import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hereditary Cancer Clinical Decision Support",
    page_icon="🔬",
    layout="wide"
)

# ============================================================
# CUSTOM STYLING (Editorial Clinical Aesthetic)
# Rich bespoke palette: Deep Navy, Turkish Teal, Crimson/Maroon, Forest Green
# Clean Off-White Canvas - NOT generic AI dark mode
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1A2530;
    }

    /* Overall background */
    .stApp {
        background-color: #F8F9FA;
    }

    /* Professional Top Clinical Masthead */
    .clinical-header {
        background: linear-gradient(135deg, #0A2540 0%, #0F3E5D 55%, #005F73 100%);
        padding: 32px 36px;
        border-radius: 12px;
        margin-bottom: 28px;
        box-shadow: 0 4px 16px rgba(10, 37, 64, 0.08);
        border-left: 6px solid #0A9396;
    }
    .clinical-header h1 {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 28px;
        font-weight: 600;
        color: #FFFFFF !important;
        margin: 0 0 8px 0;
        letter-spacing: -0.3px;
    }
    .clinical-header p {
        color: #E0E7EC !important;
        font-size: 14.5px;
        margin: 0;
        line-height: 1.5;
        font-weight: 400;
    }

    /* Clinical Cards */
    .clinical-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
    }
    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #0A2540;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 8px;
    }

    /* Custom Risk Pills */
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }
    .risk-low { background-color: #EBF7EE; color: #1B5E20; border: 1px solid #C8E6C9; }
    .risk-moderate { background-color: #E0F2F1; color: #004D40; border: 1px solid #B2DFDB; }
    .risk-high { background-color: #FFF3E0; color: #B73E00; border: 1px solid #FFE0B2; }
    .risk-very-high { background-color: #FBEAEB; color: #7B1113; border: 1px solid #F5C6CB; }

    /* Alert Banner Overrides */
    .alert-banner {
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 14px;
        font-size: 14px;
        line-height: 1.5;
    }
    .alert-maroon {
        background: #FDF2F2;
        border-left: 4px solid #7B1113;
        color: #5A0C0E;
    }
    .alert-turkish {
        background: #EBF5F7;
        border-left: 4px solid #0A9396;
        color: #00464F;
    }
    .alert-green {
        background: #F0F7F2;
        border-left: 4px solid #1B5E20;
        color: #123F16;
    }

    /* Streamlit Primary Button Overrides */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0A2540 0%, #005F73 100%);
        color: #FFFFFF;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(10, 37, 64, 0.15);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0F3E5D 0%, #0A9396 100%);
        box-shadow: 0 4px 12px rgba(10, 37, 64, 0.25);
        color: #FFFFFF;
    }

    /* Form Inputs styling */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS AND DATA
# ============================================================

@st.cache_resource
def load_models():
    return joblib.load("models_and_encoders.joblib")

@st.cache_data
def load_mutation_data():
    return pd.read_csv("processed_mutation_data.csv", low_memory=False)

try:
    models = load_models()
    mutation_df = load_mutation_data()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

clf = models["clf"]
reg = models["reg"]
le_cancer = models["le_cancer"]
le_variant = models["le_variant"]
le_risk = models["le_risk"]
features = models["features"]
weight_map = models["variant_weights"]


# ============================================================
# CLINICAL MASTHEAD
# ============================================================

st.markdown("""
<div class="clinical-header">
    <h1>Clinical Decision Support: Hereditary Cancer Risk</h1>
    <p>Genomic variant assessment grounded in empirical cohort prevalence (TCGA) and multi-factor clinical history.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PATIENT & GENOMIC INPUT SECTION
# ============================================================

st.markdown('<div class="card-title">1. Patient Profile & Identified Pathogenic Variant</div>', unsafe_allow_html=True)

col_input1, col_input2, col_input3 = st.columns([1.2, 1.2, 1.2])

with col_input1:
    patient_name = st.text_input(
        "Patient Identifier / Name",
        value="",
        placeholder="e.g., PT-84291 or Jane Doe"
    )
    gene = st.text_input(
        "Gene Symbol",
        value="",
        placeholder="e.g., TP53, BRCA1, PTEN"
    )

with col_input2:
    mutation = st.text_input(
        "Mutation / Protein Change",
        value="",
        placeholder="e.g., p.R175H, p.R248Q"
    )
    variant_classification = st.selectbox(
        "Variant Classification",
        options=[
            "Missense_Mutation",
            "Nonsense_Mutation",
            "Frame_Shift_Del",
            "Frame_Shift_Ins",
            "Splice_Site",
            "Translation_Start_Site",
            "Nonstop_Mutation",
            "In_Frame_Del",
            "In_Frame_Ins",
            "Silent",
            "Intron",
            "3'UTR",
            "5'UTR",
            "3'Flank",
            "5'Flank",
            "RNA",
            "Targeted_Region"
        ]
    )

with col_input3:
    family_history = st.selectbox(
        "Family History of Cancer",
        options=["No", "Yes", "Unknown"],
        help="Presence of verified hereditary malignancies among 1st or 2nd degree relatives."
    )
    smoking_history = st.selectbox(
        "Smoking History Category",
        options=["Never Smoked", "Have Smoked", "Current Smoker", "Unknown"],
        help="TCGA validated category for cigarette consumption."
    )

variant_weight = weight_map.get(variant_classification, 0.5)


# ============================================================
# EPS CALCULATION ENGINE
# ============================================================

def calculate_eps(cancer_type, gene_symbol, protein_change, severity_weight):
    cancer_data = mutation_df[
        mutation_df["Cancer_Type"].astype(str).str.lower()
        == str(cancer_type).lower()
    ]
    total_patients = cancer_data["Tumor_Sample_Barcode"].nunique()
    if total_patients == 0:
        return 0.0

    matching_mutations = cancer_data[
        (cancer_data["Hugo_Symbol"].astype(str).str.upper() == gene_symbol.strip().upper())
        & (cancer_data["HGVSp_Short"].astype(str).str.lower() == protein_change.strip().lower())
    ]
    mutation_patients = matching_mutations["Tumor_Sample_Barcode"].nunique()
    mutation_frequency = mutation_patients / total_patients
    return mutation_frequency * severity_weight


# Dynamic EPS Preview Bar if Gene/Mutation Entered
if gene.strip() and mutation.strip():
    with st.expander("📊 View Cohort Mutation Prevalence (EPS by Cancer Type)", expanded=False):
        eps_preview_cols = st.columns(len(le_cancer.classes_))
        for idx, c in enumerate(le_cancer.classes_):
            val = calculate_eps(c, gene, mutation, variant_weight)
            with eps_preview_cols[idx]:
                st.metric(label=c, value=f"{val:.4f}")

st.write("")
assess_btn = st.button("Evaluate Multi-Cancer Risk Profile", type="primary", use_container_width=True)


# ============================================================
# INFERENCE & CLINICAL ASSESSMENT EXECUTION
# ============================================================

if assess_btn:
    if not gene.strip():
        st.error("Please enter a Gene Symbol to run assessment.")
        st.stop()
    if not mutation.strip():
        st.error("Please enter a Mutation / Protein Change (e.g., p.R175H).")
        st.stop()

    # Numeric encoding of clinical features
    family_history_score = 1.0 if family_history == "Yes" else (0.0 if family_history == "No" else 0.5)
    if smoking_history == "Never Smoked":
        smoking_score = 0.0
    elif smoking_history == "Current Smoker":
        smoking_score = 2.0
    elif smoking_history == "Have Smoked":
        smoking_score = 1.0
    else:
        smoking_score = 1.0

    results = []
    heatmap_results = []

    for cancer in le_cancer.classes_:
        eps = calculate_eps(cancer, gene, mutation, variant_weight)
        cancer_encoded = le_cancer.transform([cancer])[0]
        try:
            variant_encoded = le_variant.transform([variant_classification])[0]
        except ValueError:
            variant_encoded = 0

        input_data = pd.DataFrame(
            [[cancer_encoded, variant_encoded, eps, family_history_score, smoking_score]],
            columns=features
        )

        risk_prediction = clf.predict(input_data)[0]
        risk_category = le_risk.inverse_transform([risk_prediction])[0]

        # Probabilities for heatmap
        risk_probabilities = clf.predict_proba(input_data)[0]
        risk_prob_dict = {
            le_risk.inverse_transform([cls_idx])[0]: prob * 100
            for cls_idx, prob in zip(clf.classes_, risk_probabilities)
        }

        heatmap_results.append({
            "Cancer Type": cancer,
            "Low": risk_prob_dict.get("Low", 0.0),
            "Moderate": risk_prob_dict.get("Moderate", 0.0),
            "High": risk_prob_dict.get("High", 0.0),
            "Very High": risk_prob_dict.get("Very High", 0.0)
        })

        if risk_category in ["High", "Very High"]:
            est_age = reg.predict(input_data)[0]
            est_age_display = f"{round(float(est_age), 1)} yrs"
        else:
            est_age_display = "—"

        results.append({
            "Cancer Type": cancer,
            "Cohort EPS": round(eps, 4),
            "Risk Category": risk_category,
            "Est. Onset Age": est_age_display
        })

    results_df = pd.DataFrame(results)

    st.write("")
    st.markdown('<div class="card-title">2. Multi-Organ Risk Assessment Results</div>', unsafe_allow_html=True)

    # 2 Column layout: Left = Table, Right = Compact Heatmap (does NOT take full page)
    col_results, col_chart = st.columns([1.1, 0.9])

    with col_results:
        st.markdown(f"**Patient Reference:** {patient_name if patient_name else 'Unregistered Profile'} | **Variant:** `{gene.upper()} {mutation}`")
        
        # Display polished dataframe with color-coded risk
        def highlight_risk(val):
            if val == "Very High":
                return "background-color: #FBEAEB; color: #7B1113; font-weight: 700;"
            elif val == "High":
                return "background-color: #FFF3E0; color: #B73E00; font-weight: 600;"
            elif val == "Moderate":
                return "background-color: #E0F2F1; color: #004D40; font-weight: 600;"
            elif val == "Low":
                return "background-color: #EBF7EE; color: #1B5E20; font-weight: 500;"
            return ""

        if hasattr(results_df.style, 'map'):
            styled_df = results_df.style.map(highlight_risk, subset=["Risk Category"])
        else:
            styled_df = results_df.style.applymap(highlight_risk, subset=["Risk Category"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    with col_chart:
        st.markdown("**Random Forest Classification Probability Distribution (%)**")
        heatmap_df = pd.DataFrame(heatmap_results).set_index("Cancer Type")[["Low", "Moderate", "High", "Very High"]]

        # Bespoke Clinical Heatmap Colormap: Soft Sage Green -> Pale Cream -> Turkish Blue -> Deep Crimson Maroon
        clinical_cmap = LinearSegmentedColormap.from_list(
            "clinical_palette",
            ["#E8F5E9", "#E0F2F1", "#80CBC4", "#F4A261", "#C94A53", "#7B1113"]
        )

        # Compact figure size (does not dominate page)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')

        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".1f",
            cmap=clinical_cmap,
            linewidths=0.6,
            linecolor="#E2E8F0",
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Confidence (%)", "shrink": 0.8},
            annot_kws={"size": 8.5, "weight": "bold", "color": "#1A2530"},
            ax=ax
        )
        ax.set_xlabel("", fontsize=9)
        ax.set_ylabel("", fontsize=9)
        ax.tick_params(axis='x', labelsize=8.5, rotation=0, colors='#0A2540')
        ax.tick_params(axis='y', labelsize=8.5, rotation=0, colors='#0A2540')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ========================================================
    # CLINICAL TRANSLATION & RECOMMENDATIONS
    # ========================================================
    st.write("")
    st.markdown('<div class="card-title">3. Clinical Translation & Actionable Surveillance</div>', unsafe_allow_html=True)

    high_tier_cancers = results_df[results_df["Risk Category"].isin(["High", "Very High"])]
    mod_tier_cancers = results_df[results_df["Risk Category"] == "Moderate"]

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown("**Personalized Surveillance Timeline**")
        if not high_tier_cancers.empty:
            st.markdown("""
            <div class="alert-banner alert-maroon">
                <strong>High-Risk Protocol:</strong><br>
                Initiate specialized organ-specific imaging (e.g. contrast MRI / targeted ultrasonography) 10 to 15 years prior to earliest estimated onset age. Frequency: every 6 to 12 months.
            </div>
            """, unsafe_allow_html=True)
        elif not mod_tier_cancers.empty:
            st.markdown("""
            <div class="alert-banner alert-turkish">
                <strong>Enhanced Protocol:</strong><br>
                Annual clinical examinations and standard surveillance starting at age 35–40 or 5–10 years prior to family earliest presentation.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-banner alert-green">
                <strong>Baseline Population Protocol:</strong><br>
                Routine age-appropriate screening per national standard clinical guidelines.
            </div>
            """, unsafe_allow_html=True)

    with col_s2:
        st.markdown("**Cascade Family Screening Priority**")
        if not high_tier_cancers.empty or family_history_score >= 1.0:
            st.markdown(f"""
            <div class="alert-banner alert-maroon">
                <strong>Priority 1 Alert:</strong><br>
                First-degree relatives (parents, siblings, adult offspring) should be notified for targeted cascade genetic testing of <code>{gene.upper()}</code> variant.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-banner alert-turkish">
                <strong>Elective Consideration:</strong><br>
                Cascade testing for extended pedigree may be considered based on overall clinical discretion.
            </div>
            """, unsafe_allow_html=True)

    with col_s3:
        st.markdown("**Genetic Counselling & Stewardship**")
        if not high_tier_cancers.empty:
            st.markdown("""
            <div class="alert-banner alert-maroon">
                <strong>Formal Referral Advised:</strong><br>
                Comprehensive hereditary cancer consultation to discuss penetrance, lifestyle mitigations, and preventive options.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-banner alert-green">
                <strong>Standard Clinical Guidance:</strong><br>
                Routine genetic risk reassessment if pedigree cancer incidence or personal health status evolves.
            </div>
            """, unsafe_allow_html=True)

    # ========================================================
    # 4. MODEL FEATURE IMPORTANCE & DECISION DRIVERS
    # ========================================================
    st.write("")
    st.markdown('<div class="card-title">4. Model Decision Drivers & Feature Importance</div>', unsafe_allow_html=True)

    col_fi1, col_fi2 = st.columns([1.1, 0.9])

    with col_fi1:
        # Map each feature strictly in the order it was trained
        feature_label_map = {
            "Cancer_Type_Encoded": "Cancer Type",
            "Variant_Classification_Encoded": "Variant Classification",
            "EPS": "Empirical Pathogenicity Score (EPS)",
            "Family_History_Score": "Family History Score",
            "Smoking_History": "Smoking History"
        }
        ordered_labels = [feature_label_map.get(f, f) for f in features]

        f_imp = pd.DataFrame({
            "Feature": ordered_labels,
            "Importance": clf.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig_fi, ax_fi = plt.subplots(figsize=(6, 2.8))
        fig_fi.patch.set_facecolor('#FFFFFF')
        ax_fi.set_facecolor('#FFFFFF')

        # Clean hospital/clinical teal-navy palette
        bars = ax_fi.barh(f_imp["Feature"], f_imp["Importance"], color="#005F73", height=0.55, edgecolor="#0A2540")
        ax_fi.set_xlim(0, max(f_imp["Importance"]) * 1.15)
        ax_fi.tick_params(axis='y', labelsize=8.5, colors='#0A2540')
        ax_fi.tick_params(axis='x', labelsize=8, colors='#64748B')
        ax_fi.spines['top'].set_visible(False)
        ax_fi.spines['right'].set_visible(False)
        ax_fi.spines['left'].set_color('#E2E8F0')
        ax_fi.spines['bottom'].set_color('#E2E8F0')
        ax_fi.grid(axis='x', linestyle='--', alpha=0.3, color='#CBD5E1')

        for bar in bars:
            width = bar.get_width()
            ax_fi.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.3f}",
                       va='center', ha='left', fontsize=8, color='#0A2540', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        plt.close(fig_fi)

    with col_fi2:
        st.dataframe(
            f_imp.sort_values("Importance", ascending=False).style.format({"Importance": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # 5. PATIENT ASSESSMENT INPUTS SUMMARY
    # ========================================================
    st.write("")
    st.markdown('<div class="card-title">5. Patient Assessment Inputs Summary</div>', unsafe_allow_html=True)

    patient_summary = pd.DataFrame({
        "Parameter / Clinical Feature": [
            "Patient Identifier / Name",
            "Gene Symbol",
            "Mutation / Protein Change",
            "Variant Classification",
            "Functional Severity Weight",
            "Family History of Cancer",
            "Smoking History Category"
        ],
        "Input Value": [
            patient_name if patient_name else "Unregistered",
            gene.upper(),
            mutation,
            variant_classification,
            f"{variant_weight:.2f}",
            family_history,
            smoking_history
        ]
    })

    st.dataframe(patient_summary, use_container_width=True, hide_index=True)

# Footer
st.write("")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 12.5px; border-top: 1px solid #E2E8F0; padding-top: 18px; margin-top: 36px;">
    <strong>Investigational Clinical Decision Support Prototype</strong> • Grounded on TCGA Pan-Cancer Data • Not an independent diagnostic device
</div>
""", unsafe_allow_html=True)