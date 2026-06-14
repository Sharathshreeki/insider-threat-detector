import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

from pipeline import run_pipeline
from utils import save_uploaded_file, list_saved_files

# ─── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Insider Threat Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
    }

    /* Risk badge colors */
    .badge-critical {
        background-color: #ff000033;
        color: #ff6b6b;
        border: 1px solid #ff000055;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #ff6b0033;
        color: #ffa94d;
        border: 1px solid #ff6b0055;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #ffd70033;
        color: #ffe066;
        border: 1px solid #ffd70055;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #00ff0022;
        color: #69db7c;
        border: 1px solid #00ff0044;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }

    /* Risk table styling */
    .risk-table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid #30363d; }
    .risk-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        color: #e6edf3;
        table-layout: fixed;
    }
    .risk-table th {
        background-color: #1c2128;
        color: #8b949e;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.05em;
        padding: 10px 12px;
        border-bottom: 1px solid #30363d;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .risk-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #21262d;
        vertical-align: middle;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .risk-table tr:last-child td { border-bottom: none; }
    .risk-table tr:hover td { background-color: #1c2128; }

    /* Fixed column widths */
    .risk-table th:nth-child(1), .risk-table td:nth-child(1) { width: 90px;  }  /* user_id */
    .risk-table th:nth-child(2), .risk-table td:nth-child(2) { width: 110px; }  /* department */
    .risk-table th:nth-child(3), .risk-table td:nth-child(3) { width: 100px; }  /* job_title */
    .risk-table th:nth-child(4), .risk-table td:nth-child(4) { width: 120px; }  /* privilege_level */
    .risk-table th:nth-child(5), .risk-table td:nth-child(5) { width: 90px;  }  /* risk_score */
    .risk-table th:nth-child(6), .risk-table td:nth-child(6) { width: 100px; }  /* risk_level */
    .risk-table th:nth-child(7), .risk-table td:nth-child(7) { width: 110px;  }  /* failed_logins */
    .risk-table th:nth-child(8), .risk-table td:nth-child(8) { width: 120px; }  /* export_count */
    .risk-table th:nth-child(9), .risk-table td:nth-child(9) { width: 140px; }  /* high_sensitivity */
    .risk-table th:nth-child(10),.risk-table td:nth-child(10){ 
    width: 280px; 
    white-space: normal; 
    word-wrap: break-word;
    line-height: 1.5;
    padding: 10px 12px;
} /* reason */
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed #30363d;
        border-radius: 8px;
        padding: 8px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #161b22; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────

RISK_COLORS = {
    "CRITICAL": "#ff6b6b",
    "HIGH":     "#ffa94d",
    "MEDIUM":   "#ffe066",
    "LOW":      "#69db7c"
}

PLOT_THEME = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="monospace"),
    xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d")
)


def risk_badge(level):
    cls = f"badge-{level.lower()}"
    return f'<span class="{cls}">{level}</span>'


def render_summary_metrics(results):
    total     = len(results)
    critical  = (results["risk_level"] == "CRITICAL").sum()
    high      = (results["risk_level"] == "HIGH").sum()
    anomalies = results.get("anomaly_score", pd.Series([])).gt(0).sum() \
                if "anomaly_score" in results.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Users Scanned", total)
    c2.metric("🔴 Critical Risk",        int(critical))
    c3.metric("🟠 High Risk",            int(high))
    c4.metric("🤖 ML Anomalies",         int(anomalies))


def render_risk_table(results):
    st.markdown('<div class="section-header">📋 User Risk Report</div>',
                unsafe_allow_html=True)

    # Filter controls
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        dept_options = ["All"] + sorted(results["department"].dropna().unique().tolist())
        dept_filter = st.selectbox("Department", dept_options)
    with col2:
        level_options = ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
        level_filter = st.selectbox("Risk Level", level_options)
    with col3:
        search = st.text_input("Search User ID", placeholder="e.g. USR001")

    filtered = results.copy()
    if dept_filter  != "All":    filtered = filtered[filtered["department"]  == dept_filter]
    if level_filter != "All":    filtered = filtered[filtered["risk_level"]  == level_filter]
    if search.strip():           filtered = filtered[filtered["user_id"].str.contains(
                                                        search.strip(), case=False, na=False)]

    # Render table with badges
    display_df = filtered[[
        "user_id", "department", "job_title",
        "privilege_level", "risk_score", "risk_level",
        "failed_logins", "export_count", "high_sensitivity", "reason"
    ]].copy()

    display_df["risk_score"] = display_df["risk_score"].round(1)

    display_df["risk_level"] = display_df["risk_level"].apply(
        lambda x: risk_badge(x)
    )

    html_table = display_df.to_html(
        escape=False,
        index=False,
        classes="risk-table",
        border=0
    )
    st.markdown(
        f'<div class="risk-table-wrap">{html_table}</div>',
        unsafe_allow_html=True
    )

    # Download button
    csv = filtered.drop(columns=["risk_level"], errors="ignore") \
                  .to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results CSV",
        csv,
        file_name="risk_report.csv",
        mime="text/csv"
    )


def render_charts(results):
    st.markdown('<div class="section-header">📊 Risk Analytics</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Risk Distribution", "Department View",
        "Top Risk Users", "Anomaly Scores"
    ])

    # ── Tab 1: Risk level pie ────────────────────────────────────────────
    with tab1:
        risk_counts = results["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            hole=0.45,
            title="Risk Level Breakdown"
        )
        fig.update_layout(**PLOT_THEME)
        fig.update_traces(textfont_color="#e6edf3")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Department heatmap ────────────────────────────────────────
    with tab2:
        dept_risk = results.groupby(["department", "risk_level"]) \
                           .size().reset_index(name="count")

        fig = px.bar(
            dept_risk,
            x="department",
            y="count",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            barmode="stack",
            title="Risk Distribution by Department"
        )
        fig.update_layout(**PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Top 10 risk users ─────────────────────────────────────────
    with tab3:
        top10 = results.nlargest(10, "risk_score")[
            ["user_id", "risk_score", "risk_level", "department"]
        ]

        fig = px.bar(
            top10,
            x="risk_score",
            y="user_id",
            orientation="h",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            title="Top 10 Highest Risk Users",
            text="risk_score"
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="monospace"),
    xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(autorange="reversed", gridcolor="#30363d")
)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Anomaly score scatter ─────────────────────────────────────
    with tab4:
        if "anomaly_score" in results.columns:
            fig = px.scatter(
                results,
                x="access_count",
                y="anomaly_score",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                hover_data=["user_id", "department", "failed_logins"],
                title="Anomaly Score vs Access Count",
                size="risk_score",
                size_max=20
            )
            fig.update_layout(**PLOT_THEME)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Anomaly scores not available.")


def render_upload_history():
    st.markdown('<div class="section-header">🗂️ Upload History</div>',
                unsafe_allow_html=True)

    files = list_saved_files()
    if not files:
        st.info("No files uploaded yet.")
        return

    history_df = pd.DataFrame(files)[["name", "size_kb", "modified"]]
    history_df.columns = ["File Name", "Size (KB)", "Uploaded At"]
    st.dataframe(history_df, use_container_width=True, hide_index=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ Insider Threat Detector")
    st.markdown("---")
    st.markdown("### Upload Data")

    logs_file  = st.file_uploader("Access Logs CSV",  type=["csv"], key="logs")
    users_file = st.file_uploader("User Profiles CSV", type=["csv"], key="users")

    run_btn = st.button("🚀 Run Detection", use_container_width=True,
                        type="primary", disabled=not (logs_file and users_file))

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "Combines rule-based scoring with IsolationForest ML "
        "to detect insider threats from user activity logs."
    )

# ─── Main ───────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ Insider Threat Detection Dashboard")
st.markdown("Upload access logs and user profiles to identify high-risk users.")
st.markdown("---")

if run_btn and logs_file and users_file:

    with st.spinner("Running detection pipeline..."):
        try:
            # Save files for future use
            save_uploaded_file(logs_file,  subfolder="logs")
            logs_file.seek(0)
            save_uploaded_file(users_file, subfolder="users")
            users_file.seek(0)

            # Read CSVs
            logs_df  = pd.read_csv(StringIO(logs_file.read().decode("utf-8")))
            users_df = pd.read_csv(StringIO(users_file.read().decode("utf-8")))

            # Run pipeline
            results = run_pipeline(logs_df, users_df)

            st.session_state["results"] = results
            st.success(f"✅ Detection complete — {len(results)} users analysed.")

        except FileNotFoundError as e:
            st.error(f"❌ Model file not found: {e}. Make sure isolation_forest.pkl and scaler.pkl are in the project folder.")
        except KeyError as e:
            st.error(f"❌ Missing column in CSV: {e}. Check that your CSV has the required columns.")
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")

# ─── Results ────────────────────────────────────────────────────────────────

if "results" in st.session_state:
    results = st.session_state["results"]

    render_summary_metrics(results)
    st.markdown("---")

    main_tab, history_tab = st.tabs(["📊 Analysis", "🗂️ Upload History"])

    with main_tab:
        render_charts(results)
        st.markdown("---")
        render_risk_table(results)

    with history_tab:
        render_upload_history()

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 80px 0; color: #8b949e;">
        <div style="font-size: 64px;">🛡️</div>
        <div style="font-size: 20px; margin-top: 16px;">No analysis run yet</div>
        <div style="font-size: 14px; margin-top: 8px;">
            Upload your Access Logs and User Profiles CSV in the sidebar, then click Run Detection.
        </div>
    </div>
    """, unsafe_allow_html=True)
