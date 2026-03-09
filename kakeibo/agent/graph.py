"""浪費防止・予算案エージェント用 LangGraph（共通＋分岐＋予算案 Yes/No ループ）. """
from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from kakeibo.db import session_scope
from kakeibo.models import ExpenseCategory
from kakeibo.services.budget_service import (
    get_budget_status,
    get_current_year_month,
)
from kakeibo.services.expense_service import (
    get_recent_expenses,
    sum_expenses_by_category_for_month,
)


# ----- 状態（拡張: 予算案フロー用） -----
class AgentState(TypedDict, total=False):
    user_id: int
    messages: str
    remaining_budget: str
    recent_expenses: str
    agent_response: str
    flow: str  # "purchase" | "budget_proposal"
    proposal_text: str
    proposal_amounts: dict  # category value str -> int (JSON 化可能)
    user_response: str
    user_feedback: str


# カテゴリ表示名（エージェント用）
CATEGORY_LABELS = {
    ExpenseCategory.food: "食費",
    ExpenseCategory.living: "生活費",
    ExpenseCategory.fixed: "固定費",
    ExpenseCategory.misc: "雑費",
    ExpenseCategory.friend: "交際費",
    ExpenseCategory.dating: "デート費",
    ExpenseCategory.hobby: "趣味費",
    ExpenseCategory.travel: "旅行費",
    ExpenseCategory.other: "その他",
}

# ルート判定: 予算案作成キーワード
BUDGET_PROPOSAL_KEYWORDS = ("予算案", "予算を作成", "予算を振り分けて", "カテゴリ別予算")


# ----- ノード: fetch_budget -----
def _node_fetch_budget(state: AgentState) -> dict:
    """残り予算・今月の支出合計を取得して state に載せる."""
    user_id = state["user_id"]
    with session_scope() as session:
        st = get_budget_status(session, user_id)
    lines = [
        f"残り日数（次の給与日まで）: {st['remaining_days']}日",
        f"今月の支出合計: {st['spent_yen']:,}円",
        f"目標額（給与日まで）: {st['target_amount_yen']:,}円",
    ]
    if st.get("available_yen") is not None:
        lines.append(f"使用可能金額: {st['available_yen']:,}円")
    else:
        lines.append("使用可能金額: 未設定（予算を設定してください）")
    return {"remaining_budget": "\n".join(lines)}


# ----- ノード: fetch_expenses -----
def _node_fetch_expenses(state: AgentState) -> dict:
    """直近 N 件の支出を取得して state に載せる."""
    user_id = state["user_id"]
    with session_scope() as session:
        expenses = get_recent_expenses(session, user_id, limit=10)
    if not expenses:
        return {"recent_expenses": "直近の支出はありません。"}
    lines = []
    for e in expenses:
        label = CATEGORY_LABELS.get(e.category, e.category.value)
        lines.append(f"・{e.spent_on.isoformat()} {e.amount_yen:,}円（{label}）{f' - {e.memo}' if e.memo else ''}")
    return {"recent_expenses": "\n".join(lines)}


# ----- ルート: 購入希望 vs 予算案作成 -----
def _route(state: AgentState) -> str:
    """messages から次のノードを決定."""
    msg = (state.get("messages") or "").strip().lower()
    for kw in BUDGET_PROPOSAL_KEYWORDS:
        if kw in msg:
            return "budget_proposal"
    return "purchase_advice"


# ----- Phase3-3: 浪費防止用プロンプト -----
AGENT_SYSTEM_PROMPT = """あなたは家計の相談に乗るアシスタントです。
ユーザーが「〇〇を買いたい」「〇〇買っていい？」と聞いたとき、次の情報を踏まえて判断し、日本語で簡潔に答えてください。

【判断に使う情報】
1. 残り予算・使用可能金額（以下に「残り予算」として渡します）
2. 直近の支出一覧（以下に「直近の支出」として渡します）

【回答のルール】
- 「許可する」「おすすめしません」など、結論をはっきり書く。
- 理由を1〜3文で述べる（残り予算〇円、今月の〇〇費は〇円なので、など）。
- 説教せず、フレンドリーに。
- 予算未設定の場合は「まず予算を設定すると、残りいくら使えるか分かります」と案内する。
"""


# ----- ノード: purchase_advice（浪費防止） -----
def _make_purchase_advice_node(llm: Any):
    def _node(state: AgentState) -> dict:
        user_msg = state.get("messages") or ""
        remaining = state.get("remaining_budget") or "（取得できていません）"
        recent = state.get("recent_expenses") or "（取得できていません）"
        system = (
            AGENT_SYSTEM_PROMPT
            + "\n\n【残り予算】\n"
            + remaining
            + "\n\n【直近の支出】\n"
            + recent
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"ユーザーの質問: {user_msg}"),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        return {"agent_response": content, "flow": "purchase"}

    return _node


# ----- 予算案: 構造化出力スキーマ -----
class CategoryBudgetProposal(BaseModel):
    """カテゴリ別予算案（LLM 構造化出力）. """
    food: int = Field(ge=0, description="食費（円）")
    living: int = Field(ge=0, description="生活費（円）")
    fixed: int = Field(ge=0, description="固定費（円）")
    misc: int = Field(ge=0, description="雑費（円）")
    friend: int = Field(ge=0, description="交際費（円）")
    dating: int = Field(ge=0, description="デート費（円）")
    hobby: int = Field(ge=0, description="趣味費（円）")
    travel: int = Field(ge=0, description="旅行費（円）")
    other: int = Field(ge=0, description="その他（円）")


BUDGET_PROPOSAL_SYSTEM = """あなたは家計のアドバイザーです。
使用可能金額と残り日数、今月のカテゴリ別支出を踏まえ、今月の残り期間で使う「カテゴリ別予算案」を作成してください。

ルール:
- 合計が使用可能金額を超えないこと。
- 各カテゴリは0以上。未使用のカテゴリは0でよい。
- 直近の支出傾向を参考にしつつ、バランスの取れた配分にすること。
"""


# ----- ノード: create_proposal（予算案作成のみ。提案表示は wait_for_yes_no で interrupt） -----
def _make_create_proposal_node(llm: Any):
    structured_llm = llm.with_structured_output(CategoryBudgetProposal, method="json_schema")

    def _node(state: AgentState) -> dict:
        user_id = state["user_id"]
        user_feedback = state.get("user_feedback") or ""

        with session_scope() as session:
            st = get_budget_status(session, user_id)
            year_month = get_current_year_month()
            by_cat = sum_expenses_by_category_for_month(session, user_id, year_month)

        available = st.get("available_yen") or 0
        remaining_days = st.get("remaining_days") or 0
        if available <= 0:
            return {
                "flow": "budget_proposal",
                "proposal_text": "使用可能金額が設定されていません。先に「予算設定 25 300000」のように目標額を設定してください。",
                "proposal_amounts": {},
                "user_response": "skip",
            }

        by_cat_str = "\n".join(
            f"・{CATEGORY_LABELS.get(c, c.value)}: {amt:,}円" for c, amt in by_cat.items()
        ) or "（今月の支出はまだありません）"

        if user_feedback:
            prompt = (
                f"【現在の使用可能金額】{available:,}円。残り{remaining_days}日。\n"
                f"【今月のカテゴリ別支出】\n{by_cat_str}\n\n"
                f"ユーザーの要望: {user_feedback}\n\n"
                "上記要望を反映した新しいカテゴリ別予算案を作成してください。合計は使用可能金額以内にすること。"
            )
        else:
            prompt = (
                f"【使用可能金額】{available:,}円。【残り日数】{remaining_days}日。\n"
                f"【今月のカテゴリ別支出】\n{by_cat_str}\n\n"
                "上記を踏まえ、カテゴリ別の予算案（円）を作成してください。合計は使用可能金額以内にすること。"
            )

        messages = [
            SystemMessage(content=BUDGET_PROPOSAL_SYSTEM),
            HumanMessage(content=prompt),
        ]
        proposal = structured_llm.invoke(messages)

        amounts = {
            "food": proposal.food,
            "living": proposal.living,
            "fixed": proposal.fixed,
            "misc": proposal.misc,
            "friend": proposal.friend,
            "dating": proposal.dating,
            "hobby": proposal.hobby,
            "travel": proposal.travel,
            "other": proposal.other,
        }
        total = sum(amounts.values())
        lines = ["**今月の予算案（カテゴリ別）**", ""]
        for cat_val, amt in amounts.items():
            if amt > 0:
                label = CATEGORY_LABELS.get(ExpenseCategory(cat_val), cat_val)
                lines.append(f"・{label}: {amt:,}円")
        lines.append(f"\n合計: {total:,}円（使用可能: {available:,}円）")
        lines.append("\nこの内容でよければ「はい」、変更したい場合は「いいえ」と送ってください。")
        proposal_text = "\n".join(lines)

        return {
            "flow": "budget_proposal",
            "proposal_text": proposal_text,
            "proposal_amounts": amounts,
        }

    return _node


# ----- ノード: wait_for_yes_no（提案表示して interrupt、resume で user_response 取得） -----
def _node_wait_for_yes_no(state: AgentState) -> dict:
    """提案文を表示して interrupt。resume で Yes/No を受け取り state に載せる."""
    text = state.get("proposal_text") or "（予算案がありません）"
    user_response = interrupt(text)
    return {
        "user_response": user_response if isinstance(user_response, str) else str(user_response),
    }


# ----- 条件エッジ: create_proposal の次（Yes → END, No → ask_feedback） -----
def _route_after_proposal(state: AgentState) -> str:
    """user_response を解釈。次のノード名を返す."""
    resp = (state.get("user_response") or "").strip().lower()
    if resp in ("skip",):
        return "__end__"
    yes_words = ("はい", "yes", "おっけー", "ok", "おk", "うん", "いいよ")
    no_words = ("いいえ", "no", "いや", "やめる", "変更")
    if any(r in resp for r in yes_words) or resp == "はい":
        return "__end__"
    if any(r in resp for r in no_words):
        return "ask_feedback"
    return "__end__"


# ----- ノード: ask_feedback（要望入力 interrupt → create_proposal へループ） -----
def _node_ask_feedback(state: AgentState) -> dict:
    """要望を促して interrupt。resume で user_feedback を受け取る."""
    feedback = interrupt("要望を教えてください。例: 食費を多めに、趣味費を減らして など")
    return {
        "user_feedback": feedback if isinstance(feedback, str) else str(feedback),
    }


# ----- グラフ組み立て -----
def create_budget_agent_graph(llm: Any):
    """
    共通: fetch_budget → fetch_expenses → route
    分岐1: purchase_advice → END
    分岐2: create_proposal → (interrupt) → handle_response → Yes: END, No: ask_feedback → (interrupt) → create_proposal ループ
    """
    builder = StateGraph(AgentState)

    builder.add_node("fetch_budget", _node_fetch_budget)
    builder.add_node("fetch_expenses", _node_fetch_expenses)
    builder.add_node("purchase_advice", _make_purchase_advice_node(llm))
    builder.add_node("create_proposal", _make_create_proposal_node(llm))
    builder.add_node("wait_for_yes_no", _node_wait_for_yes_no)
    builder.add_node("ask_feedback", _node_ask_feedback)

    builder.add_edge(START, "fetch_budget")
    builder.add_edge("fetch_budget", "fetch_expenses")
    builder.add_conditional_edges("fetch_expenses", _route, {
        "purchase_advice": "purchase_advice",
        "budget_proposal": "create_proposal",
    })
    builder.add_edge("purchase_advice", END)

    def _after_create_proposal(state: AgentState) -> str:
        """予算未設定で skip のときは END、それ以外は wait_for_yes_no."""
        if state.get("user_response") == "skip":
            return "__end__"
        return "wait_for_yes_no"

    builder.add_conditional_edges("create_proposal", _after_create_proposal, {
        "__end__": END,
        "wait_for_yes_no": "wait_for_yes_no",
    })
    builder.add_conditional_edges("wait_for_yes_no", _route_after_proposal, {
        "__end__": END,
        "ask_feedback": "ask_feedback",
    })
    builder.add_edge("ask_feedback", "create_proposal")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def run_agent_sync(
    compiled_graph: Any,
    user_id: int,
    user_message: str,
    thread_id: str,
    resume_value: Any = None,
) -> tuple[dict, bool, str]:
    """
    グラフを同期的に実行する。
    resume_value が None でないときは Command(resume=resume_value) で再開。

    Returns:
        (result_state, is_interrupted, interrupt_display)
        is_interrupted が True のときは interrupt 中。interrupt_display はユーザーに表示する文。
    """
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}

    if resume_value is not None:
        result = compiled_graph.invoke(Command(resume=resume_value), config=config)
    else:
        result = compiled_graph.invoke(
            {"user_id": user_id, "messages": user_message},
            config=config,
        )

    try:
        state = compiled_graph.get_state(config)
        is_interrupted = bool(state.next)
        next_nodes = list(state.next) if state.next else []
        # 次が ask_feedback なら要望入力の案内、それ以外は proposal_text
        if is_interrupted and next_nodes:
            next_names = [n if isinstance(n, str) else getattr(n, "name", str(n)) for n in next_nodes]
            if "ask_feedback" in next_names:
                interrupt_display = "要望を教えてください。例: 食費を多めに、趣味費を減らして など"
            else:
                interrupt_display = result.get("proposal_text") or "（表示する内容がありません）"
        else:
            interrupt_display = result.get("proposal_text") or ""
    except Exception:
        is_interrupted = False
        interrupt_display = ""

    return (result, is_interrupted, interrupt_display)
