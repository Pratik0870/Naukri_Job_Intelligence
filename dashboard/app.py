
# Streamlit Dashboard - India Job Market Intelligence
import os
import streamlit as st
import pandas as pd
import plotly.express as px

# page settings
st.set_page_config(
    page_title="India Job Market Intelligence",
    page_icon="📊",
    layout="wide"
)

# find data folder
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(PROJECT_FOLDER)
DATA_FOLDER = r"C:\Users\Pratik patil\Downloads\naukri-job-intelligence\data"

# ============================================================
# Load all data files
# ============================================================
@st.cache_data
def load_jobs():
    path = os.path.join(DATA_FOLDER, "naukri_final.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["skills"] = df["skills"].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else []
        )
        return df
    return pd.DataFrame()

@st.cache_data
def load_skills():
    path = os.path.join(DATA_FOLDER, "skill_demand.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()

@st.cache_data
def load_combos():
    path = os.path.join(DATA_FOLDER, "skill_combinations.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_roadmap():
    path = os.path.join(DATA_FOLDER, "career_roadmap.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# load everything
df         = load_jobs()
skill_df   = load_skills()
combo_df   = load_combos()
roadmap_df = load_roadmap()


# Sidebar - navigation and filters

st.sidebar.title("📊 Job Market Intelligence")
st.sidebar.markdown("**Data scraped from Naukri.com**")
st.sidebar.markdown("---")

# page navigation
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🎯 Skills",
    "🗺️ Cities",
    "🏢 Companies",
    "🚀 Career Roadmap",
    "🔍 Find Jobs"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")

# error message if no data 
if len(df) == 0:
    st.warning("No data found. Please run the scraper and analysis first.")
    st.stop()

# filter options
all_roles     = ["All"] + sorted(df["role"].dropna().unique().tolist())
all_cities    = ["All"] + sorted(df["city"].dropna().unique().tolist())
all_levels    = ["All"] + sorted(df["level"].dropna().unique().tolist())

chosen_role  = st.sidebar.selectbox("Role", all_roles)
chosen_city  = st.sidebar.selectbox("City", all_cities)
chosen_level = st.sidebar.selectbox("Level", all_levels)

# apply filters
filtered = df.copy()
if chosen_role  != "All": filtered = filtered[filtered["role"]  == chosen_role]
if chosen_city  != "All": filtered = filtered[filtered["city"]  == chosen_city]
if chosen_level != "All": filtered = filtered[filtered["level"] == chosen_level]


# PAGE 1: OVERVIEW

if page == "📊 Overview":
    st.title("📊 India Job Market Intelligence")
    st.markdown("**Live data scraped from Naukri.com | On 8 Data Science roles**")
    st.markdown("---")

    # show key numbers at top
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Jobs",     len(filtered))
    col2.metric("Companies",      filtered["company"].nunique())
    col3.metric("Cities",         filtered["city"].nunique())
    col4.metric("Fresher Jobs",   len(filtered[filtered["level"] == "Fresher/Junior"]))
    col5.metric("Remote Jobs",    len(filtered[filtered["work_mode"] == "Remote"]))

    st.markdown("---")

    # charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Jobs by Role")
        role_data = filtered["role"].value_counts().reset_index()
        role_data.columns = ["Role", "Count"]
        fig = px.bar(role_data, x="Role", y="Count",
                     color="Count", color_continuous_scale="Blues",
                     text="Count")
        fig.update_layout(xaxis_tickangle=-30, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Work Mode")
        mode_data = filtered["work_mode"].value_counts().reset_index()
        mode_data.columns = ["Mode", "Count"]
        fig = px.pie(mode_data, names="Mode", values="Count",
                     color_discrete_sequence=["#2ecc71", "#3498db", "#e74c3c"],
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Seniority Levels")
        level_data = filtered["level"].value_counts().reset_index()
        level_data.columns = ["Level", "Count"]
        fig = px.bar(level_data, x="Level", y="Count",
                     color="Count", color_continuous_scale="Oranges",
                     text="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Hiring Companies")
        comp_data = filtered["company"].value_counts().head(10).reset_index()
        comp_data.columns = ["Company", "Jobs"]
        fig = px.bar(comp_data.sort_values("Jobs"),
                     x="Jobs", y="Company",
                     orientation="h",
                     color="Jobs", color_continuous_scale="Purples")
        st.plotly_chart(fig, use_container_width=True)

    # insights box
    st.markdown("---")
    st.subheader("📌 Key Insights from Data")
    top_city  = filtered["city"].value_counts().index[0] if len(filtered) > 0 else "N/A"
    top_role  = filtered["role"].value_counts().index[0] if len(filtered) > 0 else "N/A"
    remote_p  = round((filtered["work_mode"] == "Remote").mean() * 100, 1)
    fresher_p = round((filtered["level"] == "Fresher/Junior").mean() * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.info(f"🏙️ **{top_city}** has most DS jobs")
    c2.info(f"💼 **{top_role}** has most openings")
    c3.info(f"🏠 Only **{remote_p}%** jobs are remote")
    c4.info(f"🎓 **{fresher_p}%** open to freshers")


# PAGE 2: SKILLS
elif page == "🎯 Skills":
    st.title("🎯 Skills Intelligence")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 Skills Employers Want")
        if len(skill_df) > 0:
            sd = skill_df.head(15).reset_index()
            sd.columns = ["Skill", "Count", "Pct"]
            fig = px.bar(sd.sort_values("Count"),
                         x="Count", y="Skill",
                         orientation="h",
                         color="Count", color_continuous_scale="Reds",
                         text="Pct")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Most Common Skill Pairs")
        if len(combo_df) > 0:
            fig = px.bar(combo_df.head(12).sort_values("count"),
                         x="count", y="combination",
                         orientation="h",
                         color="count", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

    # skill heatmap
    st.markdown("---")
    st.subheader("Which Skills Each Role Needs")
    if len(skill_df) > 0:
        top_skills = skill_df.head(10).index.tolist()
        heatmap_rows = []
        for role in filtered["role"].unique():
            role_skill_list = []
            for s in filtered[filtered["role"] == role]["skills"]:
                role_skill_list.extend(s)
            counts = pd.Series(role_skill_list).value_counts()
            row = {"Role": role}
            for skill in top_skills:
                row[skill] = counts.get(skill, 0)
            heatmap_rows.append(row)

        if heatmap_rows:
            hm = pd.DataFrame(heatmap_rows).set_index("Role")
            fig = px.imshow(hm.astype(int),
                           color_continuous_scale="Blues",
                           text_auto=True, aspect="auto")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # tips
    st.markdown("---")
    st.subheader("💡 What This Means For You")
    if len(skill_df) > 0:
        top3 = skill_df.head(3).index.tolist()
        st.success(f"✅ **Must learn first:** {', '.join(top3)}")
        st.info("💡 After basics — add Power BI or Tableau for DA roles, or Machine Learning for DS roles")
        st.warning("⚠️ Cloud skills (AWS, Azure, GCP) are growing fast — add one cloud platform")


# PAGE 3: CITIES

elif page == "🗺️ Cities":
    st.title("🗺️ City Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Jobs by City")
        city_data = filtered["city"].value_counts().head(10).reset_index()
        city_data.columns = ["City", "Count"]
        fig = px.bar(city_data, x="City", y="Count",
                     color="Count", color_continuous_scale="Greens",
                     text="Count")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Remote vs Onsite by City")
        top_cities = filtered["city"].value_counts().head(6).index
        city_mode  = filtered[filtered["city"].isin(top_cities)]
        city_mode  = city_mode.groupby(["city", "work_mode"]).size().reset_index(name="count")
        fig = px.bar(city_mode, x="city", y="count", color="work_mode",
                     barmode="group",
                     color_discrete_map={
                         "Onsite": "#2ecc71",
                         "Remote": "#3498db",
                         "Hybrid": "#e74c3c"
                     })
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 City Insights")
    top_city = filtered["city"].value_counts().index[0] if len(filtered) > 0 else "Bangalore"
    top_count = filtered["city"].value_counts().iloc[0] if len(filtered) > 0 else 0
    st.success(f"🏙️ **{top_city}** has {top_count} jobs — biggest hub for Data Science in India")
    st.info("💡 Mumbai is strong for Business Analyst and BI roles — especially in banking and finance")
    st.warning("🏠 Remote jobs are rare — only 8-10%. Consider relocating to Bangalore or Mumbai")


# PAGE 4: COMPANIES
elif page == "🏢 Companies":
    st.title("🏢 Company Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 Companies Hiring")
        comp_data = filtered["company"].value_counts().head(15).reset_index()
        comp_data.columns = ["Company", "Jobs"]
        fig = px.bar(comp_data.sort_values("Jobs"),
                     x="Jobs", y="Company",
                     orientation="h",
                     color="Jobs", color_continuous_scale="Purples",
                     text="Jobs")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Company Type")
        def company_type(name):
            n = str(name).lower()
            if any(w in n for w in ["tcs", "infosys", "wipro", "hcl", "accenture", "capgemini"]):
                return "IT Services"
            if any(w in n for w in ["google", "apple", "microsoft", "amazon"]):
                return "Big Tech"
            if any(w in n for w in ["bank", "finance", "insurance"]):
                return "BFSI"
            return "Other"

        filtered["comp_type"] = filtered["company"].apply(company_type)
        ct = filtered["comp_type"].value_counts().reset_index()
        ct.columns = ["Type", "Count"]
        fig = px.pie(ct, names="Type", values="Count", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Company Insights")
    top_comp = filtered["company"].value_counts().index[0] if len(filtered) > 0 else "TCS"
    st.success(f"🏆 **{top_comp}** is hiring the most Data roles right now")
    st.info("💡 IT services companies hire most — good for first job. Big Tech pays more but needs stronger skills")


# PAGE 5: CAREER ROADMAP
elif page == "🚀 Career Roadmap":
    st.title("🚀 Career Roadmap Engine")
    st.markdown("**Skills recommended based on what employers actually ask for in job listings**")
    st.markdown("---")

    if len(roadmap_df) > 0:
        target_role = st.selectbox(
            "What role do you want to get?",
            roadmap_df["role"].unique()
        )

        role_data = roadmap_df[roadmap_df["role"] == target_role].sort_values("priority")

        st.subheader(f"Skills you need for: {target_role}")

        # show skills as columns
        skill_list = role_data["skill"].tolist()
        cols = st.columns(len(skill_list))
        for i, (col, skill) in enumerate(zip(cols, skill_list)):
            with col:
                st.metric(f"#{i+1}", skill.title())

        st.markdown("---")
        st.subheader("Learning Order")

        for i, skill in enumerate(skill_list, 1):
            if i <= 2:
                st.success(f"✅ Step {i} — **{skill.title()}** — Learn this first")
            elif i <= 4:
                st.info(f"📚 Step {i} — **{skill.title()}** — Core skill")
            else:
                st.warning(f"⭐ Step {i} — **{skill.title()}** — Advanced — add after basics")

        # comparison table
        st.markdown("---")
        st.subheader("All Roles Comparison")
        pivot = roadmap_df.pivot_table(
            index="skill", columns="role",
            values="priority", aggfunc="min"
        ).fillna("—")
        st.dataframe(pivot, use_container_width=True)
    else:
        st.warning("Run analysis first to see career roadmap.")

# PAGE 6: JOB SEARCH
elif page == "🔍 Find Jobs":
    st.title("🔍 Find Jobs")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        search_text = st.text_input(
            "Search job title or company",
            placeholder="e.g. Data Analyst, TCS, Bangalore"
        )
    with col2:
        fresher_only = st.checkbox("Show fresher/junior jobs only")

    # apply search
    results = filtered.copy()
    if search_text:
        results = results[
            results["title"].str.contains(search_text, case=False, na=False) |
            results["company"].str.contains(search_text, case=False, na=False)
        ]
    if fresher_only:
        results = results[results["level"] == "Fresher/Junior"]

    st.markdown(f"**Found {len(results)} jobs**")

    # show table
    show_cols = ["title", "company", "city", "level", "work_mode", "role", "experience"]
    show_cols = [c for c in show_cols if c in results.columns]
    st.dataframe(results[show_cols].reset_index(drop=True), use_container_width=True)

    # download button
    csv_data = results.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv_data,
        file_name="jobs.csv",
        mime="text/csv"
    )

# footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Made by Pratik Patil**")
st.sidebar.markdown("B.Sc. Data Science | Mumbai")
st.sidebar.markdown("[GitHub Link ](https://github.com/Pratik0870)")
