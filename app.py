import os
from dotenv import load_dotenv
load_dotenv("file.env")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter
import re

st.set_page_config(page_title="JobPilot", layout="wide")

st.title("JobPilot: Smart Job Matcher & Resume Builder")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_jobs():
    return pd.read_csv("data/jobpilot_jobs_with_embedding_text_deploy.csv")

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("artifacts/job_faiss_deploy.index")

model = load_model()
jobs_df = load_jobs()
index = load_faiss_index()

st.success("JobPilot loaded successfully.")

tab1, tab2, tab3 = st.tabs([
    "Job Recommendations",
    "Batch Analytics",
    "About / Architecture"
])

with tab1:
    st.header("Job Recommendations")

    uploaded_resume = st.file_uploader(
        "Upload your resume PDF",
        type=["pdf"]
    )

    if uploaded_resume is not None:
        from pypdf import PdfReader

        def extract_text_from_uploaded_pdf(uploaded_file):
            reader = PdfReader(uploaded_file)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            return text

        resume_text = extract_text_from_uploaded_pdf(uploaded_resume)

        st.success("Resume uploaded and parsed successfully.")

        st.subheader("Resume Preview")
        st.text_area("Extracted resume text", resume_text[:3000], height=250)

        SKILL_KEYWORDS = [
            "python", "r", "sql", "pandas", "numpy", "scikit-learn", "sklearn",
            "matplotlib", "tableau", "power bi", "excel",
            "pyspark", "spark", "databricks", "mongodb", "mysql", "sqlite",
            "etl", "api", "data pipeline", "data cleaning",
            "machine learning", "logistic regression", "regression",
            "classification", "clustering", "pca", "random forest",
            "decision tree", "nlp", "sentiment analysis", "topic modeling",
            "data analytics", "business analytics", "dashboard",
            "visualization", "quality engineering", "process improvement",
            "root cause analysis", "documentation", "solidworks", "catia",
            "jira", "git", "github"
        ]

        def extract_skills(text, skill_keywords=SKILL_KEYWORDS):
            text_lower = text.lower()
            matched_skills = []

            for skill in skill_keywords:
                pattern = r"\b" + re.escape(skill.lower()) + r"\b"
                if re.search(pattern, text_lower):
                    matched_skills.append(skill)

            return sorted(set(matched_skills))

        def infer_career_level(resume_text):
            text = resume_text.lower()

            senior_terms = [
                "senior", "sr.", "staff", "principal",
                "lead", "manager", "director", "vp"
            ]

            junior_terms = [
                "student", "graduate", "junior", "jr.",
                "intern", "internship"
            ]

            for term in senior_terms:
                if term in text:
                    return "senior"

            for term in junior_terms:
                if term in text:
                    return "early"

            return "early"

        resume_skills = extract_skills(resume_text)
        career_level = infer_career_level(resume_text)

        st.subheader("Extracted Profile")
        st.write("Skills:", resume_skills)
        st.write("Career level:", career_level)

        target_roles = st.multiselect(
            "Target roles",
            ["Data Analyst", "Business Analyst", "Analytics Engineer", "Data Scientist", "Data Engineer"],
            default=["Data Analyst", "Business Analyst", "Analytics Engineer", "Data Scientist"]
        )

        preferred_locations = st.multiselect(
            "Preferred locations",
            ["Remote", "California", "Bay Area", "San Francisco", "Oakland", "San Jose", "New York", "Chicago"],
            default=["Remote", "California", "Bay Area", "San Francisco", "Oakland", "San Jose"]
        )

        user_profile = {
            "resume_text": resume_text,
            "skills": resume_skills,
            "target_roles": target_roles,
            "preferred_locations": preferred_locations,
            "career_level": career_level,
            "dealbreakers": ["contract only", "unpaid", "temporary position", "temporary role"]
        }

        def build_profile_text(user_profile):
            return (
                "Target roles: " + ", ".join(user_profile["target_roles"]) + ". " +
                "Skills: " + ", ".join(user_profile["skills"]) + ". " +
                "Preferred locations: " + ", ".join(user_profile["preferred_locations"]) + ". " +
                "Resume summary: " + user_profile["resume_text"][:3000]
            )

        def retrieve_jobs(profile_text, model, index, jobs_df, top_k=100):
            profile_embedding = model.encode(
                [profile_text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            scores, indices = index.search(profile_embedding, top_k)

            results = jobs_df.iloc[indices[0]].copy()
            results["semantic_similarity"] = scores[0]

            return results.reset_index(drop=True)

        def apply_hard_filters(candidates_df, user_profile):
            filtered = candidates_df.copy()
            career_level = user_profile.get("career_level", "early").lower()

            senior_terms = ["senior", "sr.", "sr ", "staff", "principal", "lead", "manager", "director", "vp"]
            junior_terms = ["junior", "jr.", "jr ", "entry level", "graduate", "intern", "internship"]
            contract_terms = ["contract only", "unpaid", "temporary position", "temporary role", "contract position", "contract role"]

            def violates_filter(row):
                title = str(row.get("title", "")).lower()
                description = str(row.get("description", "")).lower()
                combined_text = title + " " + description

                for term in contract_terms:
                    if term in combined_text:
                        return True

                if career_level in ["student", "early", "new_grad"]:
                    for term in senior_terms:
                        if term in title:
                            return True

                if career_level in ["senior", "experienced"]:
                    for term in junior_terms:
                        if term in title:
                            return True

                return False

            filtered["violates_dealbreaker"] = filtered.apply(violates_filter, axis=1)

            kept_jobs = filtered[filtered["violates_dealbreaker"] == False].copy()
            removed_jobs = filtered[filtered["violates_dealbreaker"] == True].copy()

            return kept_jobs.reset_index(drop=True), removed_jobs.reset_index(drop=True)

        def normalize_score(series):
            min_val = series.min()
            max_val = series.max()

            if max_val == min_val:
                return np.ones(len(series))

            return (series - min_val) / (max_val - min_val)

        def compute_skill_match(job_text, user_skills):
            job_text = str(job_text).lower()
            matched = []

            for skill in user_skills:
                pattern = r"\b" + re.escape(skill.lower()) + r"\b"
                if re.search(pattern, job_text):
                    matched.append(skill)

            if len(user_skills) == 0:
                return 0, matched

            return len(matched) / len(user_skills), matched

        def compute_role_match(title, target_roles):
            title = str(title).lower()

            for role in target_roles:
                role_lower = role.lower()

                if role_lower in title:
                    return 1.0

                role_words = role_lower.split()
                if all(word in title for word in role_words):
                    return 1.0

            return 0.0

        def compute_location_match(location, preferred_locations):
            location = str(location).lower()

            for preferred in preferred_locations:
                preferred = preferred.lower()

                if preferred in location:
                    return 1.0

                if preferred == "bay area" and any(city in location for city in ["san francisco", "oakland", "san jose", "berkeley", "palo alto"]):
                    return 1.0

                if preferred == "california" and any(ca in location for ca in ["ca", "california", "san francisco", "san jose", "oakland", "los angeles", "san diego"]):
                    return 1.0

            return 0.0

        def rank_jobs(filtered_jobs, user_profile):
            ranked = filtered_jobs.copy()

            ranked["combined_job_text"] = (
                ranked["title"].fillna("") + " " +
                ranked["description"].fillna("")
            )

            skill_results = ranked["combined_job_text"].apply(
                lambda text: compute_skill_match(text, user_profile["skills"])
            )

            ranked["skill_match_score"] = skill_results.apply(lambda x: x[0])
            ranked["matched_skills"] = skill_results.apply(lambda x: x[1])

            ranked["role_match_score"] = ranked["title"].apply(
                lambda title: compute_role_match(title, user_profile["target_roles"])
            )

            ranked["location_match_score"] = ranked["location"].apply(
                lambda location: compute_location_match(location, user_profile["preferred_locations"])
            )

            ranked["semantic_score_norm"] = normalize_score(ranked["semantic_similarity"])

            ranked["final_score"] = (
                0.50 * ranked["semantic_score_norm"] +
                0.25 * ranked["skill_match_score"] +
                0.15 * ranked["role_match_score"] +
                0.10 * ranked["location_match_score"]
            )

            return ranked.sort_values("final_score", ascending=False).reset_index(drop=True)

        def generate_explanation(job_row):
            parts = []

            if len(job_row["matched_skills"]) > 0:
                parts.append("Matched skills: " + ", ".join(job_row["matched_skills"]))

            if job_row["role_match_score"] > 0:
                parts.append("Matches a target role")

            if job_row["location_match_score"] > 0:
                parts.append("Matches preferred location")

            parts.append(f"Semantic similarity: {job_row['semantic_similarity']:.3f}")
            parts.append(f"Final score: {job_row['final_score']:.3f}")

            return " | ".join(parts)

        if st.button("Find Matching Jobs"):
            profile_text = build_profile_text(user_profile)

            retrieved_jobs = retrieve_jobs(
                profile_text=profile_text,
                model=model,
                index=index,
                jobs_df=jobs_df,
                top_k=100
            )

            filtered_jobs, removed_jobs = apply_hard_filters(retrieved_jobs, user_profile)
            ranked_jobs = rank_jobs(filtered_jobs, user_profile)

            ranked_jobs["explanation"] = ranked_jobs.apply(generate_explanation, axis=1)

            st.session_state["ranked_jobs"] = ranked_jobs
            st.session_state["resume_text"] = resume_text

            st.success(f"Retrieved 100 jobs, filtered to {len(filtered_jobs)}, and ranked recommendations.")

        if "ranked_jobs" in st.session_state:
            ranked_jobs = st.session_state["ranked_jobs"]
            resume_text = st.session_state["resume_text"]

            st.subheader("Top Recommended Jobs")

            display_cols = [
                "title", "company", "location", "final_score",
                "semantic_similarity", "matched_skills", "explanation", "job_url"
            ]

            if "adaptive_score" in ranked_jobs.columns:
                display_cols.insert(4, "adaptive_score")


            st.dataframe(ranked_jobs[display_cols].head(20))

            csv = ranked_jobs[display_cols].head(20).to_csv(index=False)

            st.download_button(
                label="Download Top 20 Jobs as CSV",
                data=csv,
                file_name="jobpilot_top_jobs.csv",
                mime="text/csv"
            )

            st.subheader("Explain & Generate Resume")

            job_options = [
                f"{i}: {row['title']} at {row['company']} ({row['location']})"
                for i, row in ranked_jobs.head(20).iterrows()
            ]

            selected_option = st.selectbox(
                "Select a job",
                job_options
            )

            selected_idx = int(selected_option.split(":")[0])
            selected_job = ranked_jobs.loc[selected_idx]

            with st.expander("Why this job was recommended", expanded=True):
                st.write(selected_job["explanation"])
                st.write("Final score:", round(selected_job["final_score"], 3))
                st.write("Semantic similarity:", round(selected_job["semantic_similarity"], 3))
                st.write("Matched skills:", selected_job["matched_skills"])

            feedback = st.radio(
                "Feedback on this recommendation",
                ["Accept", "Reject", "Skip"],
                horizontal=True
            )

            st.session_state[f"feedback_{selected_idx}"] = feedback

            st.write("Selected feedback:", feedback)

            # -----------------------------
            # Adaptive Learning from Feedback
            # -----------------------------

            if "ranking_weights" not in st.session_state:
                st.session_state["ranking_weights"] = {
                    "semantic": 0.50,
                    "skill": 0.25,
                    "role": 0.15,
                    "location": 0.10
                }

            def apply_feedback_learning(ranked_jobs, selected_idx, feedback):
                updated_jobs = ranked_jobs.copy()
                weights = st.session_state["ranking_weights"].copy()

                selected_row = updated_jobs.loc[selected_idx]

                if feedback == "Accept":
                    weights["skill"] += 0.05 * selected_row["skill_match_score"]
                    weights["role"] += 0.05 * selected_row["role_match_score"]
                    weights["location"] += 0.05 * selected_row["location_match_score"]

                elif feedback == "Reject":
                    if selected_row["skill_match_score"] < 0.15:
                        weights["skill"] += 0.05

                    if selected_row["role_match_score"] == 0:
                        weights["role"] += 0.05

                    if selected_row["location_match_score"] == 0:
                        weights["location"] += 0.03

                    weights["semantic"] -= 0.03

                elif feedback == "Skip":
                    weights["semantic"] -= 0.01
                    weights["skill"] += 0.01

                # Prevent negative weights
                for key in weights:
                    weights[key] = max(weights[key], 0.01)

                # Normalize weights
                total = sum(weights.values())
                weights = {k: v / total for k, v in weights.items()}

                updated_jobs["adaptive_score"] = (
                    weights["semantic"] * updated_jobs["semantic_score_norm"] +
                    weights["skill"] * updated_jobs["skill_match_score"] +
                    weights["role"] * updated_jobs["role_match_score"] +
                    weights["location"] * updated_jobs["location_match_score"]
                )

                updated_jobs = (
                    updated_jobs
                    .sort_values("adaptive_score", ascending=False)
                    .reset_index(drop=True)
                )

                st.session_state["ranking_weights"] = weights

                return updated_jobs, weights

            if st.button("Apply Feedback Learning"):
                before_rank = selected_idx + 1

                updated_jobs, updated_weights = apply_feedback_learning(
                    ranked_jobs,
                    selected_idx,
                    feedback
                )

                selected_job_id = selected_job["job_id"]

                after_rank_list = updated_jobs.index[
                    updated_jobs["job_id"] == selected_job_id
                ].tolist()

                if len(after_rank_list) > 0:
                    after_rank = after_rank_list[0] + 1
                else:
                    after_rank = None

                st.session_state["ranked_jobs"] = updated_jobs

                st.success("Feedback applied. Recommendations have been re-ranked.")

                st.write("Updated ranking weights:")
                st.json(updated_weights)

                if after_rank is not None:
                    st.write(f"Selected job rank before feedback: {before_rank}")
                    st.write(f"Selected job rank after feedback: {after_rank}")

                    if after_rank < before_rank:
                        st.write("The selected job moved higher in the ranking.")
                    elif after_rank > before_rank:
                        st.write("The selected job moved lower in the ranking.")
                    else:
                        st.write("The selected job stayed in the same rank position.")                                
            

        def generate_tailored_resume_streamlit(resume_text, selected_job):
            import os
            from google import genai
            import time

            api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

            if not api_key:
                return "Missing GEMINI_API_KEY. Add your Gemini API key as an environment variable or Streamlit secret."

            client = genai.Client(api_key=api_key)

            prompt = f"""
You are an expert resume writer.

Tailor the candidate's resume to the selected job.

IMPORTANT RULES:
- Do NOT invent employers.
- Do NOT invent degrees.
- Do NOT invent dates.
- Do NOT invent projects.
- Do NOT invent platform-specific tools like Azure, AWS, or GCP unless they are explicitly present in the resume.
- Only rewrite and emphasize existing experience.
- Improve keyword alignment with the job description.
- Keep the output professional and ATS-friendly.

TARGET JOB:
Title: {selected_job['title']}
Company: {selected_job['company']}
Location: {selected_job['location']}
Job Description: {str(selected_job['description'])[:4000]}

ORIGINAL RESUME:
{resume_text[:6000]}

OUTPUT FORMAT:
1. Professional Summary
2. Key Skills
3. Tailored Experience Bullets
4. Why This Resume Fits The Job
"""

            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    return response.text
                except Exception as e:
                    time.sleep(5 * (attempt + 1))

            return "Resume generation failed. Please try again."

        if st.button("Generate Tailored Resume"):
            with st.spinner("Generating tailored resume..."):
                tailored_resume = generate_tailored_resume_streamlit(
                    resume_text,
                    selected_job
                )

            st.session_state["tailored_resume"] = tailored_resume

        if "tailored_resume" in st.session_state:
            st.subheader("Tailored Resume Draft")
            st.text_area(
                "Generated resume",
                st.session_state["tailored_resume"],
                height=500
            )

            st.download_button(
                label="Download Tailored Resume as TXT",
                data=st.session_state["tailored_resume"],
                file_name="jobpilot_tailored_resume.txt",
                mime="text/plain"
            )

with tab2:
    st.header("Batch Analytics")

    ANALYTICS_SKILLS = [
        "python", "sql", "r", "excel", "tableau", "power bi",
        "databricks", "spark", "pyspark", "aws", "azure", "gcp",
        "machine learning", "nlp", "regression", "clustering",
        "pandas", "numpy", "scikit-learn", "etl", "api",
        "dashboard", "visualization"
    ]

    def count_skill_mentions(jobs_df, skills=ANALYTICS_SKILLS):
        counts = Counter()

        for text in jobs_df["description"].fillna("").astype(str):
            text_lower = text.lower()

            for skill in skills:
                pattern = r"\b" + re.escape(skill.lower()) + r"\b"
                if re.search(pattern, text_lower):
                    counts[skill] += 1

        return (
            pd.DataFrame(counts.items(), columns=["skill", "job_count"])
            .sort_values("job_count", ascending=False)
            .reset_index(drop=True)
        )

    top_skills_df = count_skill_mentions(jobs_df)

    top_locations_df = (
        jobs_df["location"]
        .value_counts()
        .reset_index()
    )
    top_locations_df.columns = ["location", "job_count"]

    top_companies_df = (
        jobs_df["company"]
        .value_counts()
        .reset_index()
    )
    top_companies_df.columns = ["company", "job_count"]

    top_titles_df = (
        jobs_df["title"]
        .value_counts()
        .reset_index()
    )
    top_titles_df.columns = ["title", "job_count"]

    salary_summary = {
        "Total jobs": len(jobs_df),
        "Jobs with salary_min": jobs_df["salary_min"].notna().sum(),
        "Jobs with salary_max": jobs_df["salary_max"].notna().sum(),
        "Jobs with salary_text": jobs_df["salary_text"].fillna("").astype(str).str.strip().ne("").sum()
    }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Skills")
        st.dataframe(top_skills_df.head(15))
        st.bar_chart(top_skills_df.head(10).set_index("skill"))

        st.subheader("Top Companies")
        st.dataframe(top_companies_df.head(15))

    with col2:
        st.subheader("Top Locations")
        st.dataframe(top_locations_df.head(15))
        st.bar_chart(top_locations_df.head(10).set_index("location"))

        st.subheader("Top Titles")
        st.dataframe(top_titles_df.head(15))

    st.subheader("Salary Data Coverage")
    st.json(salary_summary)

with tab3:
    st.header("About / Architecture")

    st.markdown("""
    JobPilot is a multi-stage job recommendation and resume tailoring system.

    **Pipeline:**

    1. Ingest and deduplicate job postings  
    2. Extract resume text and skills  
    3. Generate dense embeddings for jobs and user profiles  
    4. Retrieve candidates using FAISS  
    5. Apply profile-aware hard filters  
    6. Re-rank jobs using semantic similarity, skill overlap, role match, and location match  
    7. Explain why each job was recommended  
    8. Generate a tailored resume for a selected job  

    **BAX-423 techniques used:**
    - Embedding-based retrieval
    - FAISS nearest-neighbor search
    - Multi-stage recommendation pipeline
    - Re-ranking
    - Adaptive feedback learning
    """)