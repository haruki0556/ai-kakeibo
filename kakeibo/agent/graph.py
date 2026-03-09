"""浪費防止エージェント用 LangGraph（Phase3-1, 3-3）. """
from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from kakeibo.db import session_scope
from kakeibo.models import ExpenseCategory
from kakeibo.services.budget_service import get_budget_status
from kakeibo.services.expense_service import get_recent_expenses


# ----- 状態（計画どおり） -----
class AgentState(TypedDict, total=False):
    user_id: int
    messages: str
    remaining_budget: str
    recent_expenses: str
    agent_response: str


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


# ----- ノード: fetch_budget（ツール的・DB参照） -----
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


# ----- ノード: fetch_expenses（ツール的・DB参照） -----
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


# ----- Phase3-3: エージェント用システムプロンプト -----
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


# ----- ノード: agent（LLM で許可/非許可と理由を生成） -----
def _make_agent_node(llm: Any):
    def _node_agent(state: AgentState) -> dict:
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
        return {"agent_response": content}

    return _node_agent


# ----- グラフ組み立て: route → fetch_budget → fetch_expenses → agent → END -----
def create_budget_agent_graph(llm: Any):
    """
    Phase3-1: LangGraph で浪費防止エージェントを定義。
    エッジ: START → fetch_budget → fetch_expenses → agent → END
    """
    builder = StateGraph(AgentState)

    builder.add_node("fetch_budget", _node_fetch_budget)
    builder.add_node("fetch_expenses", _node_fetch_expenses)
    builder.add_node("agent", _make_agent_node(llm))

    builder.add_edge(START, "fetch_budget")
    builder.add_edge("fetch_budget", "fetch_expenses")
    builder.add_edge("fetch_expenses", "agent")
    builder.add_edge("agent", END)

    return builder.compile()


def run_agent_sync(compiled_graph: Any, user_id: int, user_message: str) -> str:
    """
    グラフを同期的に実行し、agent_response を返す。
    app から asyncio.to_thread で呼ぶ想定。
    """
    result = compiled_graph.invoke(
        {"user_id": user_id, "messages": user_message},
        config={"configurable": {}},
    )
    return result.get("agent_response") or "判断できませんでした。"