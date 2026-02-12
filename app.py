"""
Bug Hunter Agent - Streamlit UI
-------------------------------
Interactive frontend for the Agentic RAG Bug Detector.
"""

import streamlit as st
import pandas as pd
import asyncio
import os
import json
from io import BytesIO
from bug_agent import BugHunterAgent, MCP_SERVER_URL, HF_MODEL_ID

# Page Config
st.set_page_config(
    page_title="Bug Hunter Agent",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Hacker/Dark" theme
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #00ff41; 
        font-family: 'Courier New', Courier, monospace;
    }
    h2, h3 {
        color: #e6e6e6;
    }
    .stButton>button {
        color: #0e1117;
        background-color: #00ff41;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00cc33;
        color: #0e1117;
    }
    .stTextInput>div>div>input {
        color: #00ff41;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🕷️ Configuration")
model_id = st.sidebar.text_input("HF Model ID", value=HF_MODEL_ID)
server_url = st.sidebar.text_input("MCP Server URL", value=MCP_SERVER_URL)
st.sidebar.markdown("---")
st.sidebar.markdown("**Agent Status:**")
if st.sidebar.button("Check Connectivity"):
    try:
        # Simple connectivity check logic here if needed
        st.sidebar.success("Agent Configured")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Main Content
st.title("🕷️ Bug Hunter Agent")
st.markdown(f"**Agentic RAG Bug Detector** powered by `{model_id}` and `CppCheck`")

tab1, tab2, tab3 = st.tabs(["🧩 Single Snippet Analysis", "📂 Batch CSV Processing", "🔍 Results Inspector"])

# --- TAB 1: Single Snippet ---
with tab1:
    st.header("Analyze Code Snippet")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        context_input = st.text_area("Context / Intent", height=150, placeholder="Describe what the code is supposed to do...")
    with col2:
        code_input = st.text_area("C++ Code", height=300, placeholder="Paste code here...", value="""// Example
void test() {
    int x;
    int array[10];
    array[10] = 0; 
}""")

    if st.button("Analyze Snippet", type="primary"):
        if not context_input or not code_input:
            st.warning("Please provide both context and code.")
        else:
            with st.spinner("🕷️ Hunting bugs... (RAG Search -> CppCheck -> LLM Analysis)"):
                try:
                    # Initialize agent
                    agent = BugHunterAgent(server_url)
                    
                    # Run async analysis
                    result = asyncio.run(agent.analyze_single_snippet(code_input, context_input))
                    
                    # Display Results
                    st.success("Analysis Complete!")
                    
                    # 1. Bugs Found
                    bugs = result.get("Bug Line")
                    if bugs:
                        st.error(f"❌ Bugs Detected at line(s): {bugs}")
                        st.markdown(f"**Explanation:** {result.get('Explanation')}")
                    else:
                        st.success("✅ No bugs detected")
                        st.markdown(f"**Explanation:** {result.get('Explanation')}")
                    
                    # 2. Corrected Code (Diff View)
                    st.markdown("### 🛠️ Corrected Code Suggestion")
                    corrected_code = result.get("Corrected Code")
                    if corrected_code:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.caption("Original")
                            st.code(code_input, language="cpp")
                        with col_b:
                            st.caption("Suggested Fix")
                            st.code(corrected_code, language="cpp")
                    else:
                        st.info("No corrected code generated (or no bugs found).")
                        
                    # 3. Evidence
                    with st.expander("🔍 View Evidence (RAG Docs & Static Analysis)"):
                        st.markdown("**RAG Documents Retrieved:**")
                        rag_docs = result.get("RAG Docs", [])
                        if rag_docs:
                            for i, doc in enumerate(rag_docs):
                                st.markdown(f"**Doc {i+1}** (Score: {doc.get('score', 0):.3f})")
                                st.code(doc.get('text', '')[:300] + "...", language="text")
                        else:
                            st.warning("No relevant documentation found.")
                            
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

# --- TAB 2: Batch Processing ---
with tab2:
    st.header("Batch CSV Processing")
    uploaded_file = st.file_uploader("Upload CSV (ID, Context, Code)", type="csv", key="batch_upload")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.write(f"Total entries: {len(df)}")
        
        if st.button("Run Batch Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a placeholder for results
            results = []
            agent = BugHunterAgent(server_url)
            
            # Processing loop
            # Note: In a real app, we might want to run this in a separate thread/process
            # but for simplicity we run it here with UI updates
            
            output_placeholder = st.empty()
            
            async def run_batch():
                # Connect once
                from fastmcp import Client
                async with Client(server_url) as mcp_client:
                    for index, row in df.iterrows():
                        status_text.text(f"Analyzing ID {row['ID']} ({index+1}/{len(df)})...")
                        
                        res = await agent.analyze_entry(
                            mcp_client, 
                            str(row['ID']), 
                            str(row.get('Code', '')), 
                            str(row.get('Context', ''))
                        )
                        
                        # Clean result for CSV
                        csv_res = res.copy()
                        csv_res.pop("RAG Docs", None)
                        results.append(csv_res)
                        
                        # Update progress
                        progress_bar.progress((index + 1) / len(df))
                        
                        # Show intermediate output
                        output_placeholder.dataframe(pd.DataFrame(results).tail(3))
                        
            # Run the batch
            asyncio.run(run_batch())
            
            status_text.success("Batch Processing Complete!")
            
            # Final Results
            final_df = pd.DataFrame(results)
            st.dataframe(final_df)
            
            # Download Button
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="bug_hunter_results.csv",
                mime="text/csv",
            )

# --- TAB 3: Results Inspector ---
with tab3:
    st.header("🔍 Results Inspector (Drill Down)")
    st.markdown("Select an ID to view the full details, including original code, bugs, and suggested fixes.")

    col1, col2 = st.columns(2)
    with col1:
        in_file = st.file_uploader("1. Upload Input CSV (with Code)", type="csv", key="inspect_in")
    with col2:
        out_file = st.file_uploader("2. Upload Output CSV (Results)", type="csv", key="inspect_out")

    # Try to load local defaults if not uploaded
    if not in_file and os.path.exists("data/input.csv"):
        st.info("Using local `data/input.csv`")
        in_df = pd.read_csv("data/input.csv")
        in_file = True # Flag as loaded
    elif in_file:
        in_df = pd.read_csv(in_file)
    
    if not out_file and os.path.exists("data/final_output.csv"):
        st.info("Using local `data/final_output.csv`")
        out_df = pd.read_csv("data/final_output.csv")
        out_file = True # Flag as loaded
    elif not out_file and os.path.exists("data/output.csv"):
         st.info("Using local `data/output.csv`")
         out_df = pd.read_csv("data/output.csv")
         out_file = True
    elif out_file:
        out_df = pd.read_csv(out_file)

    if in_file and out_file:
        # Merge DataFrames on ID
        try:
            # Ensure ID is string for merging
            in_df['ID'] = in_df['ID'].astype(str)
            out_df['ID'] = out_df['ID'].astype(str)
            
            merged_df = pd.merge(in_df, out_df, on="ID", how="inner")
            
            st.success(f"Loaded {len(merged_df)} entries successfully.")
            
            # Selection Dropdown
            selected_id = st.selectbox("Select ID to Inspect:", merged_df['ID'].unique())
            
            if selected_id:
                row = merged_df[merged_df['ID'] == selected_id].iloc[0]
                
                st.markdown("---")
                st.subheader(f"Analyzing ID: {selected_id}")
                
                # Context
                st.markdown("**Context / Intent:**")
                st.info(row.get('Context', 'N/A'))
                
                # Bug Details
                bug_lines = row.get('Bug Line')
                if pd.notna(bug_lines) and str(bug_lines).strip():
                    st.error(f"❌ Bugs Detected at line(s): {bug_lines}")
                    st.markdown(f"**Explanation:** {row.get('Explanation')}")
                else:
                    st.success("✅ No bugs detected.")
                
                # Code Diff View
                st.markdown("### 🛠️ Code Comparison")
                
                orig_code = row.get('Code', '')
                fix_code = row.get('Corrected Code', '')
                
                if pd.isna(fix_code):
                     fix_code = ""

                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("Original Code")
                    st.code(orig_code, language="cpp")
                with col_b:
                    st.caption("Corrected Code")
                    if fix_code:
                        st.code(fix_code, language="cpp")
                    else:
                        st.caption("(No correction available)")

        except KeyError as e:
            st.error(f"Error merging CSVs: Missing column {e}. Make sure both CSVs have an 'ID' column.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please upload both Input and Output CSVs to inspect results.")
