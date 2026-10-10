---
layout: post
comments: true
title: "When a $1.5 Million Claim Depends on an X-Ray, a PDF, and a Phone Call: Building a Multi-Modal Insurance Lakehouse on Google Cloud"
published: true
---

magine you are a senior claims examiner at a life insurance company on a Monday morning.

A **$1,500,000 Whole Life insurance claim** lands on your desk. The customer, a 62-year-old executive named **Robert Sterling**, bought the policy just **seven months ago**. When he applied, he qualified for the company’s cheapest "Preferred Plus" rate by signing a health declaration stating:

> *"I have never been diagnosed with heart disease, cardiomyopathy, or an irregular heartbeat, and I do not have any implanted medical devices."*

Now, seven months later, he has tragically passed away from sudden cardiac arrest.

If you only look at your company’s traditional database—the rows and columns of numbers—everything looks fine. His premiums were paid on time. The hospital’s billing code simply says "Cardiac Arrest."

Wait. Because the policy is less than two years old, it falls inside what the insurance industry calls the **24-Month Contestability Window**—the grace period where the insurer has the right to check whether the customer told the truth on their application.

Where is the truth hiding? It isn't in a spreadsheet. It is scattered across **three completely different types of files** sitting in three different computer systems:
1. **A Hospital Chest X-Ray (`PNG`)** taken in the emergency room—where a trained eye (or a vision AI) can spot the unmistakable metal outline of an **implanted heart pacemaker** that has been sitting in his chest for years.
2. **A 4-Page Doctor’s Statement (`PDF`)** faxed by the hospital—which mentions in paragraph three that Robert was first diagnosed with severe heart failure four years *before* he bought the insurance policy.
3. **A Recorded Phone Call (`WAV` Audio)** from the call center—where Robert’s grieving wife casually mentions: *"We immediately called his regular cardiologist, Dr. Grant, who has been checking Robert’s pacemaker for the past four years."*

---

## 1. The Real Pain Point in Insurance: The "Four-Silo" Data Maze

If you work in an insurance company today—whether in Life, Health, Critical Illness, or Property & Casualty—you know how painful it is to connect those four clues.

In most enterprises, **structured data** (tables of policy numbers, dates, and dollar amounts) lives in a data warehouse. Meanwhile, **unstructured data**—the X-rays, the PDF contracts, the doctor's notes, and the call center voice recordings—lives in a chaotic maze of file servers, buckets, and third-party tools.

![Breaking the Insurance Data Maze: Siloed Pipelines vs. Google Cloud Multi-Modal Lakehouse](images/202610/01_siloed_vs_lakehouse_architecture.png)

Historically, if an engineering team wanted AI to check all four clues together, they had to build a fragile "Rube Goldberg machine":
- **System 1**: A SQL database for the policy numbers.
- **System 2**: An Optical Character Recognition (OCR) pipeline to scrape text out of PDF files (often scrambling tables and medical dates along the way).
- **System 3**: A Speech-to-Text pipeline to transcribe phone calls.
- **System 4**: A separate "Vector Database" (like Pinecone or Milvus) to store and search medical images.

Moving sensitive medical files across four different systems is slow, expensive, and a nightmare for privacy compliance. As a result, two bad things happen in the real world:
1. **Honest families wait weeks** for legitimate payouts while human examiners manually open PDFs and listen to phone recordings.
2. **Sophisticated fraud slips through the cracks** because nobody has time to cross-check every X-ray against every phone call.

What if we could bring the tables, the X-rays, the PDFs, and the phone calls **under one single roof**—and ask questions across all of them at once?

That is the idea behind **OmniClaim Life Lakehouse**, an open-source project we built on **Google Cloud**.

---

## 2. What Is a "Multi-Modal Lakehouse" in Plain English?

Think of a traditional data warehouse as a **librarian who can only read spreadsheets**. If you hand that librarian a chest X-ray or an audio recording of a phone call, they have no idea what to do with it.

Google Cloud’s **BigQuery Lakehouse** upgrades that librarian into a **multi-modal detective** using four simple building blocks:

![The 4-Act Detective Story: How the Lakehouse Connects Every Clue for One Customer](images/202610/02_four_act_customer_storyline.png)

1. **Apache Iceberg Tables (For the Numbers & Dates)**:
   Instead of locking policy data inside a proprietary format, we store our structured policy and claim records as open-standard **Apache Iceberg** tables (`PARQUET` files) directly in our private Google Cloud Storage bucket.
2. **BigLake Object Tables (A Catalog for Images, PDFs, and Audio)**:
   Without moving or copying a single file, BigQuery creates an "Object Table"—essentially a live index of every `.png` X-ray, `.pdf` document, and `.wav` voice recording sitting in our cloud storage bucket.
3. **In-Engine Vector Search (`gemini-embedding-2-preview`)**:
   BigQuery uses Google’s newest multimodal embedding model to turn every medical X-ray into a "visual fingerprint" (a list of 3,072 numbers that capture what the lungs, heart, and bones look like). Using `VECTOR_SEARCH`, BigQuery can instantly compare a new claim X-ray against a healthy baseline X-ray—or against thousands of past claims to spot stolen images.
4. **Native AI Reading & Listening (`Gemini 3.8 Flash` via `ObjectRef`)**:
   Instead of running separate OCR or Speech-to-Text software, BigQuery hands a secure pointer (`ObjectRef`) directly to **Gemini 3.8 Flash**. Inside a standard SQL query, Gemini *reads* the PDF contracts and *listens* to the `.wav` phone call at the same time.

---

## 3. Real Medical Data, Not Fake Stick Figures

When we started building this Proof-of-Concept, we made a strict rule: **no fake computer-drawn X-rays and no robotic 1990s computer voices**. If an insurance leader or medical underwriter looks at a demo, the data has to look and sound like Monday morning at a real carrier.

- **Real Clinical X-Rays from the US National Library of Medicine (NIH)**: Every chest and trauma radiograph in our project comes from peer-reviewed medical case reports in the **NIH Open-i / PubMed Central & MedPix repository**.
- **Lifelike Human Voices (`Google Cloud Text-to-Speech Chirp 3: HD`)**: Every beneficiary phone call (`.wav`) is synthesized using Google Cloud’s newest `Chirp 3: HD` conversational voices—complete with natural pauses, hesitations, and emotional inflection.

Let’s walk through **three real-world insurance mysteries** from our dataset to see how the Google Cloud Lakehouse solves them.

---

## 4. Three Real-World Insurance Mysteries Solved by the Lakehouse

### Mystery #1: The $1.5 Million Concealed Pacemaker (`CLM-LIFE-1002`)

Let’s return to **Robert Sterling**, the customer from the beginning of our story who bought a $1.5M Whole Life policy claiming zero heart history and passed away **7 months later**.

Look at what happens when BigQuery compares three real NIH radiographs side-by-side in **Act II (Visual X-Ray Forensics)**:

![Case Study #1: Robert Sterling ($1.5M Whole Life) — The Concealed Pacemaker Mystery](images/202610/03_robert_sterling_pacemaker_comparison.png)

- **Panel 1 (Left)** shows a normal, healthy chest X-ray (`NIH PMC12127709`)—what a "Preferred Plus" customer’s chest should look like when they buy a policy.
- **Panel 2 (Center)** is Robert’s emergency room X-ray (`NIH PMC7033380, Panel A`). Even if you aren’t a radiologist, look at the upper-left side of his chest (the right side of the image): there is a circular metal **cardiac pacemaker generator** with bright white wires snaking directly into a massively enlarged heart!
- **Panel 3 (Right)** is the reference image BigQuery’s `VECTOR_SEARCH` automatically matched (`NIH PMC7033380, Panel B`, with **93.55% visual similarity**)—a known clinical profile of chronic Stage III heart failure with a biventricular pacemaker implanted years earlier.

Next, inside the exact same BigQuery pipeline, **Gemini 3.8 Flash** reads the doctor’s PDF statement and listens to the wife’s `.wav` phone call. Both confirm that Robert’s pacemaker was implanted **nearly three years before he signed his insurance application**.

Instead of accidentally paying out $1,500,000 on a fraudulent application, the Lakehouse automatically flags the claim for **`SIU_RESCISSION_HOLD` ($0 Payout)**—saving the carrier **$1.5 million**.

---

### Mystery #2: The Pixel-Tweaked Stolen X-Ray Fraud Ring (`CLM-CI-1004`)

Critical Illness insurance pays a lump-sum cash benefit (for example, $350,000) the moment a policyholder is diagnosed with a serious condition like lung cancer, helping them pay for treatment while they are still alive.

Because these policies pay out cash quickly upon proof of diagnosis, organized fraud rings sometimes steal a real cancer patient’s X-ray from an earlier claim and submit it under a brand-new policy.

To beat traditional fraud detection, the fraudster opens the stolen X-ray image on their computer and **changes the brightness of a single pixel in the corner by +1**.

Why? Because traditional insurance systems check for duplicate files using a cryptographic file hash (like `MD5` or `SHA-256`). Changing even one single pixel scrambles the entire hash code (`7961cd93...` becomes `5e2739c7...`), tricking the computer into thinking it is a brand-new medical file!

![Case Study #2: Victor Krum ($350K Critical Illness) — Catching a Recycled X-Ray Fraud Ring](images/202610/04_victor_krum_duplicate_xray_fraud.png)

Here is how **BigQuery `VECTOR_SEARCH`** stops this fraud ring cold:
- Unlike a file hash that looks at raw bytes, **`gemini-embedding-2-preview`** looks at the **visual anatomy** of the image—the shape of the ribs, the curve of the diaphragm, and the Left Upper Lobe lung mass (`NIH MedPix MPX1219`).
- When BigQuery compares Victor Krum’s "new" X-ray against our historical claims atlas, it spots a **99.62% Cosine Similarity (`distance = 0.0038`)** to Claim `PRIOR-CLM-8821`, which was submitted 9 months ago under a completely different customer’s name!
- Combined with Gemini 3.8 Flash noticing that the caller on the `.wav` audio refuses to name his treating oncologist, the Lakehouse immediately blocks the payout (`SIU_FRAUD_RING_REJECT`, **$350,000 saved**).

---

### Mystery #3: The $2 Million Mountaineering Exclusion—Fairness for the Family (`CLM-ADD-1003`)

AI in insurance shouldn't just be about catching fraud; it should be about **fairness and accuracy**.

Consider **Marcus Thorne** (`CLM-ADD-1003`). Marcus held a **$1,000,000 Base Term Life policy** plus an optional **$1,000,000 Accidental Death & Dismemberment (AD&D) Double-Indemnity Rider**. He had paid his premiums faithfully for **38 months**—well past the 24-month Contestability Window.

When Marcus died in a tragic fall, the paper claim form submitted to the insurer said: *"Accidental slip on a recreational walking trail"*—requesting the full **$2,000,000** double payout.

![Case Study #3: Marcus Thorne ($2M Claim) — Resolving a Paper-vs-Audio Rider Conflict](images/202610/06_marcus_thorne_mountaineering_exclusion.png)

When the Lakehouse examines all four clues together, a nuanced story emerges:
1. **The Trauma X-Ray (`NIH PMC5702695`)**: BigQuery Vector Search matches a high-energy displaced collarbone fracture (`0.9547` similarity) typical of a severe vertical fall, not a flat walking trail slip.
2. **The Coroner PDF & Policy Contract PDF**: Gemini 3.8 Flash reads both PDFs side-by-side. The medical examiner notes that Mountain Rescue recovered Marcus at **4,250 meters on Liberty Ridge, Mount Rainier**. Meanwhile, **Section 4.2(a) of the AD&D Rider contract** explicitly excludes *"unroped free-solo alpine climbing above 4,000 meters."*
3. **The Brother’s Phone Call (`.wav` Audio)**: On the recorded call, Marcus’s brother David speaks honestly: *"Even though the paper form mentioned a walking trail, I want to be honest with you—Marcus was doing an unroped free-solo ice climb at 4,250 meters on Mount Rainier."*

What is the right decision here?
- Because the policy is **38 months old** (incontestable), the **$1,000,000 Base Life benefit** is 100% valid and must be paid promptly to his grieving family.
- However, because the accident occurred during an excluded free-solo climb above 4,000 meters, the **$1,000,000 AD&D Rider** is denied under Section 4.2(a).

Gemini 3.8 Flash synthesizes all of these rules in seconds and recommends **`PARTIAL_APPROVE_BASE_DENY_ADD_RIDER` ($1,000,000 Approved / $1,000,000 Rider Denied)**.

---

## 5. How Simple Is the Code? A Peek Under the Hood

You don't need to write thousands of lines of Python glue code to connect these four modalities. In Google Cloud BigQuery, the entire workflow happens in **three short, readable SQL steps**:

### Step A: Expose Unstructured Files in Cloud Storage as an Object Table
```sql
CREATE OR REPLACE EXTERNAL TABLE `your-gcp-project-id.omniclaim_lakehouse.obj_claim_radiographs`
WITH CONNECTION `your-gcp-project-id.us-central1.lakehouse-omniclaim-conn`
OPTIONS (
    object_metadata = 'SIMPLE',
    uris = ['gs://your-gcp-project-id-bucket/unstructured_evidence/claim_radiographs/*.png']
);
```

### Step B: Search X-Ray Images Using `VECTOR_SEARCH`
```sql
SELECT
    query.claim_id,
    base.ref_image_id AS matched_ref_image_id,
    ROUND(1.0 - distance, 4) AS cosine_similarity
FROM VECTOR_SEARCH(
    TABLE `your-gcp-project-id.omniclaim_lakehouse.ref_medical_atlas_embeddings`, 'embedding',
    TABLE `your-gcp-project-id.omniclaim_lakehouse.claims_image_embeddings`, 'embedding',
    top_k => 1,
    distance_type => 'COSINE'
);
```

### Step C: Let Gemini 3.8 Flash Read PDFs & Listen to Audio Directly from GCS
Notice how `OBJ.MAKE_REF` passes the raw PDF or `.wav` audio file directly from Google Cloud Storage into `ML.GENERATE_TEXT`—no OCR or Speech-to-Text pipeline required!
```sql
SELECT
    claim_id,
    ml_generate_text_llm_result AS pdf_clinical_contract_audit
FROM ML.GENERATE_TEXT(
    MODEL `your-gcp-project-id.omniclaim_lakehouse.model_gemini_auditor`, -- gemini-3.8-flash
    (
        SELECT
            c.claim_id,
            STRUCT(
                CONCAT('Audit the Policy PDF and Physician APS PDF for ', c.claim_id) AS prompt,
                [
                    OBJ.FETCH_METADATA(OBJ.MAKE_REF(c.policy_pdf_gcs_uri, 'your-gcp-project-id.us-central1.lakehouse-omniclaim-conn')),
                    OBJ.FETCH_METADATA(OBJ.MAKE_REF(c.aps_pdf_gcs_uri, 'your-gcp-project-id.us-central1.lakehouse-omniclaim-conn'))
                ] AS object_ref_list
            ) AS prompt
        FROM `your-gcp-project-id.omniclaim_lakehouse.iceberg_claims_policies` c
    ),
    STRUCT(0.0 AS temperature, 1024 AS max_output_tokens, TRUE AS flatten_json_output)
);
```

---

## 6. The Big Picture: Results Across All 5 Hero Cases

When we ran all **5 Hero Life & Living Benefits Claims** ($4,600,000 in total claimed benefits) through our live Google Cloud Lakehouse pipeline, the system achieved **5 out of 5 exact visual pathology matches** (`0.9355` to `1.0000` cosine similarity) and **5 out of 5 accurate payout decisions**:

![Live Google Cloud Lakehouse Results: 5 Hero Claims ($4.60M Total Exposure)](images/202610/05_portfolio_adjudication_dashboard.png)

Look at the two-sided business impact:
- **$1,750,000 Fast-Tracked to Honest Families**:
  - **Eleanor Vance (`CLM-LIFE-1001`)**: Passed away from acute bacterial pneumonia (`NIH PMC10405870`) after holding her policy for 68 clean months $\rightarrow$ **$500,000 Auto-Approved**.
  - **Marcus Thorne (`CLM-ADD-1003`)**: Incontestable Base Life benefit honored $\rightarrow$ **$1,000,000 Approved**.
  - **Diana Chen (`CLM-CI-1005`)**: Non-smoking policyholder diagnosed with a 3.0 cm Stage IIA lung adenocarcinoma (`NIH PMC7529567`), verified via pathology biopsy PDF $\rightarrow$ **$250,000 Living Benefit Approved** so she can begin cancer treatment immediately.
- **$2,850,000 Protected Against Fraud & Excluded Riders**:
  - **Robert Sterling (`CLM-LIFE-1002`)**: Undisclosed pacemaker & 4-year cardiomyopathy inside the 24-month Contestability Window $\rightarrow$ **$1,500,000 Saved**.
  - **Marcus Thorne (`CLM-ADD-1003` AD&D Rider)**: Free-solo climbing above 4,000m exclusion enforced $\rightarrow$ **$1,000,000 Saved**.
  - **Victor Krum (`CLM-CI-1004`)**: Pixel-tweaked stolen lung cancer X-ray caught by Vector Search $\rightarrow$ **$350,000 Saved**.

---

## 7. Explore the Open-Source Project & Interactive Workbench

We built an interactive, step-by-step **Streamlit Examiner Workbench (`app.py`)** where you can click through each of the 5 customer storylines, inspect the 3-way NIH X-ray comparisons, read the physician & policy PDFs, play the `Chirp 3: HD` beneficiary phone recordings, and view the underlying BigQuery SQL.

The entire project is open-sourced on GitHub:
👉 **[https://github.com/crownpku/omniclaims_lakehouse_multimodal](https://github.com/crownpku/omniclaims_lakehouse_multimodal)**

You can run the interactive demo locally on your laptop in under 15 seconds (even before setting up a Google Cloud project) with just three commands:

```bash
git clone https://github.com/crownpku/omniclaims_lakehouse_multimodal.git
cd omniclaims_lakehouse_multimodal
uv sync
uv run streamlit run app.py
```

Whether you are a **cloud architect exploring BigLake Apache Iceberg, Object Tables, and Vertex AI Gemini 3.8 Flash**, or an **insurance leader looking to modernize claims adjudication and SIU fraud detection**, the era of siloed data is over. Your spreadsheets, your images, your PDFs, and your audio recordings can finally speak the same language.
