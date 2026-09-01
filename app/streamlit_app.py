from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="科研论文多模态证据检索", layout="wide")
st.title("科研论文多模态证据检索")
st.caption("答案 → 证据森林 → 原句/原图 → PDF 页码与坐标")

api_url = st.sidebar.text_input("API URL", os.getenv("PAPER_RAG_API_URL", "http://localhost:8000"))
question = st.text_area("问题", placeholder="达到 500 MPa 拉伸强度的材料或结构有哪些？")

if st.button("检索并回答", type="primary", disabled=not question.strip()):
    with st.spinner("检索闭合证据森林……"):
        try:
            response = requests.post(
                f"{api_url.rstrip('/')}/query",
                json={"query": question, "answer_type": "entity_list"},
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"API 调用失败：{exc}")
        else:
            st.subheader("回答")
            st.write(result.get("answer") or "当前部署未启用生成模型。")
            st.caption(f"证据成本：{result.get('total_cost')}")
            st.subheader("证据森林")
            for component in result.get("forest", []):
                with st.expander(
                    f"论文 {component['paper_id']} · {len(component['node_ids'])} 个证据节点"
                ):
                    for evidence in component.get("evidence", []):
                        st.markdown(
                            f"**{evidence['node_type']} · {evidence['node_id']} · "
                            f"第 {evidence.get('page') or '?'} 页**"
                        )
                        if evidence.get("text"):
                            st.write(evidence["text"])
                        image_path = evidence.get("image_path")
                        if image_path and os.path.exists(image_path):
                            st.image(image_path)
                        st.caption(
                            f"bbox={evidence.get('bbox')} · "
                            f"confidence={evidence.get('confidence')}"
                        )
