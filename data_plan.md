RomCoM-X: Data Preparation and Augmentation Agent Specification
Version: 1.0
Project: RomCoM-X
Owner: Project owner (human)
Agent assignment: Data discovery, provenance, cleaning, split construction, romanization inventory, augmentation, and data-quality assurance only.
Target language for training: Hindi-English (Hindi written in Latin script)
Target tasks: Natural language inference (NLI), paraphrase detection, and toxicity detection.
Target experimental data conditions: canonical, realistic romanization variation, and matched random-noise control.  
0. Mission and final handoff
Prepare a reproducible, auditable, Hindi-only dataset package for the RomCoM-X experiments. The package must contain the source records, source-level split manifests, an attested Hindi romanization inventory, and three matched training/validation dataset conditions:
1. EXP-01 Canonical: eligible Hindi tokens normalized to a deterministic canonical Romanized form derived from the Dakshina Hindi training lexicon.
2. EXP-02 Realistic: eligible Hindi tokens probabilistically replaced with alternative attested Romanizations from the same lexicon.
3. EXP-03 Matched noise: the same source rows and eligible token positions receive artificial character perturbations matched to the edit magnitude of EXP-02.
The original, unaugmented source dataset must also be retained as the immutable reference.

The final handoff must make it possible for the project owner to open the resulting files, inspect the reports, and independently proceed to model work without repeating the data preparation.

The agent must not begin model training or model evaluation. The assignment ends when the prepared datasets, manifests, tests, and required data-quality reports are complete and the final readiness report is issued.
1. Strict scope boundary
1.1 In scope
- Locate and verify source datasets and their official releases.
- Verify file access, schemas, revisions, licenses, attribution requirements, and usage restrictions.
- Download only the required data; do not download large archives unnecessarily.
- Preserve raw sources and calculate checksums.
- Inspect source schemas, nulls, malformed records, label distributions, script composition, duplicates, and source identifiers.
- Obtain the correct Hindi-only task-labelled source records for NLI, paraphrase, and toxicity.
- Normalize records into a consistent structured format without changing their meaning or labels.
- Build source-level train/validation splits where an official training split does not already provide them.
- Construct and audit a Hindi Romanization variant inventory using Dakshina.
- Inspect COMI-LINGUA as a supplementary Hinglish reference, without treating it as a substitute task-label dataset.
- Implement a deterministic, frequency-informed Romanization augmentation pipeline.
- Generate canonical, realistic, and matched-noise data conditions.
- Run data-only validation: schema checks, label preservation, protected-token checks, split leakage checks, augmentation trace checks, and coverage summaries.
- Produce the reports and machine-readable manifests specified in this document.
1.2 Explicitly out of scope
Do not perform any of the following:
- Base-model validation, Qwen/PEFT compatibility checks, GPU smoke tests, or model loading.
- Baseline model inference or evaluation.
- QLoRA, LoRA, SFT, or any other model training.
- Model-performance analysis, accuracy comparisons, benchmark scoring, statistical significance tests, bootstrap analysis, or error analysis of model predictions.
- Prompt engineering for model evaluation.
- Model selection or fallback-model selection.
- W&B experiment runs or training-run tracking.
- Repository packaging, release preparation, model cards, README writing, publication writing, or resume copy.
- Publishing or uploading any dataset, adapter, or model artifact.
- Adding new languages to training. Bengali, Gujarati, and Tamil remain held-out languages for the planned experiments.
- Inventing new NLI/paraphrase/toxicity labels or silently substituting COMI-LINGUA labels.
- Generating new code-switch structures. The augmentation changes Romanized Hindi spellings only; it does not create Hindi-English switches or alter the existing switch locations.
- Using a language model to rewrite, paraphrase, translate, or relabel source data.
Data QA is in scope. Model evaluation is not. The agent must calculate data-quality and augmentation-coverage statistics, but it must not run a model or report model performance.
2. Frozen project decisions
Use these project decisions unless the project owner explicitly changes them.
Item	Decision
Project	RomCoM-X
Training language	Hindi-English only
Held-out transfer languages	Bengali-English, Gujarati-English, Tamil-English
Tasks	NLI, paraphrase detection, toxicity detection
Primary transfer task in the larger project	Toxicity detection
Backbone	Not part of this assignment; do not validate or load it
Augmentation	Hindi Romanization variation only
Code-mixing	Reuse code-mixed structures already present in the approved task data
Experimental data conditions	Canonical, realistic, matched random noise
Labels	Preserve source labels and task-specific target structures exactly
Split unit	Original source example, not augmented row
Augmentation generation	Deterministic, offline, saved to disk; no stochastic changes during later training
Training examples per condition	Same source IDs, same row count, same labels, one generated view per source record


The core research question is whether Hindi-only training with realistic Romanized spelling variation improves understanding of Romanized Indic-English code-mixed inputs and whether any improvement transfers to held-out Indic languages. Data preparation must not introduce additional experimental variables.
3. Source roles and rules
3.1 Indi-RomCoM: primary task-labelled source, subject to access verification
Paper: https://arxiv.org/abs/2606.30790
HTML paper: https://arxiv.org/html/2606.30790

The paper describes a fine-tuning corpus of 395,076 instances across seven tasks, four Indic languages, and three code-mixing intensities, with 80% of instances per task stratified by code-mixing level. The paper's underlying task sources include:
- Paraphrase: IndicXParaphrase.
- NLI / inference: IndicXNLI.
- Toxicity: IndicAlign-Toxic, supplemented with benign samples from IndicAlign-Instruct.
The paper describes task-specific preparation and code-mixing rules. The actual downloadable training artifact, exact files, revision, and license must still be verified independently. Do not assume that a described training split is publicly downloadable or redistributable.
Required access procedure
1. Inspect the paper, its appendices, author/project pages, official repositories, Hugging Face organizations, supplementary materials, and any official data links.
2. Search for an author-released Indi-RomCoM training dataset or the exact fine-tuning split.
3. Record every candidate URL, repository ID, revision/commit, file name, file size, access result, license, and whether the source is official.
4. Verify that the files actually contain task-labelled records and not only the public evaluation benchmark.
5. Verify schema, task, language, mixing-intensity, source IDs, train/evaluation partition, and label structure.
6. Filter only the Hindi training records for NLI, paraphrase, and toxicity.
7. Preserve task-specific fields and target formats. Do not flatten paired tasks into a single generic text field if doing so loses premise/hypothesis or sentence-pair structure.
8. Preserve original source IDs and links to the original base prompt wherever available.
9. Keep all related variants of one underlying source example in the same split.
10. Exclude all benchmark evaluation examples from training, augmentation-inventory construction, and training-validation split selection.
Important task-specific handling
The paper describes special treatment for toxicity: the harmful source prompt is kept intact, while the classification/refusal output is code-mixed. The agent must inspect the real released schema and record exactly which fields are inputs, targets, explanations, labels, and protected content.
- Never alter the toxicity label.
- Never alter or paraphrase the harmful source content.
- Never change refusal requirements, structured output markers, or target schema.
- Do not assume the toxicity input contains Romanized Hindi. Report the number and percentage of examples with eligible Romanized Hindi in each input field and target field separately.
- Augmentation is intended to improve understanding of Romanized Hindi input. Therefore, only transform approved input text fields. Do not modify target answers or labels to manufacture apparent input coverage.
- If toxicity examples have no eligible Romanized Hindi in their input fields, report that limitation clearly. Do not silently augment targets or change the experimental objective. Mark this as a project-owner decision gate.
For NLI and paraphrase, preserve both sides of each pair and the exact relation label. The agent may transform eligible Hindi tokens in both input-side text fields, but must not change the relation or reorder the pair.
If no official training split is accessible
Do not silently substitute COMI-LINGUA or fabricate labels.
1. Produce a blocker report with all official locations searched, dates, URLs, and results.
2. Identify the three original task datasets named by the paper and inspect their Hindi train partitions and licenses as a possible fallback:
   - IndicXNLI for NLI.
   - IndicXParaphrase for paraphrase.
   - IndicAlign-Toxic plus approved benign IndicAlign-Instruct examples for toxicity.
3. Specifically verify that the claimed IndicAlign toxicity configuration exists in the current repository and inspect its actual schema. Do not assume a configuration name from a paper or URL is available.
4. Treat the previously observed license issues as unresolved until checked against the current canonical dataset card and repository:
   - IndicXNLI has had conflicting license metadata/card information (CC0 metadata versus CC BY-NC 4.0 card). Treat as restricted / unresolved until reconciled.
   - IndicXParaphrase has had no clear license declared in the inspected metadata. Treat as unresolved until the owner/license is established.
   - IndicAlign license and the specific toxicity/instruction subsets must be verified at the source.
5. Do not create code-mixed text from these source datasets unless the owner approves a faithful recreation of the paper's original transformation procedure. The data agent may prepare a source-readiness report, but must not invent its own code-switch generator.
6. Stop before final dataset generation if the training source, label schema, or license basis remains unclear. Ask for the project owner's decision through the final blocker report.
3.2 Dakshina: primary Romanization inventory source
Official repository: https://github.com/google-research-datasets/dakshina
Dataset paper: https://www.aclweb.org/anthology/2020.lrec-1.294
Dataset license stated by repository: CC BY-SA 4.0.

Dakshina is a collection of native-script and Latin-script text for South Asian languages. Its Hindi lexicon contains human-annotated native-word/Romanization pairs and a count of attestations for each pair. The count is an annotation-attestation count, not a real-world typing probability.
Files and permitted roles
Hindi file	Role in this project	Training-inventory use?
hi/lexicons/hi.translit.sampled.train.tsv	Main word-level variant inventory	Yes
hi/lexicons/hi.translit.sampled.dev.tsv	Lexicon quality diagnostics only	No
hi/lexicons/hi.translit.sampled.test.tsv	Lexicon quality diagnostics only	No
hi/romanized/hi.romanized.rejoined.dev.native.txt	Held-out sentence-level data QA	No
hi/romanized/hi.romanized.rejoined.dev.roman.txt	Held-out sentence-level data QA	No
hi/romanized/hi.romanized.rejoined.test.native.txt	Held-out sentence-level data QA	No
hi/romanized/hi.romanized.rejoined.test.roman.txt	Held-out sentence-level data QA	No
hi/romanized/hi.romanized.rejoined.tsv	Combined manually romanized validation sample; not training data	No
hi/romanized/hi.romanized.rejoined.aligned.tsv	Derived token alignment for the sentence sample	No
hi/romanized/hi.romanized.rejoined.aligned.cased_nopunct.tsv	Derived aligned representation	No
hi/romanized/hi.romanized.split.tsv and validation files	Diagnostic/derivative data only	No


The Dakshina README states that full-sentence romanization prompts were sampled from the native-script Wikipedia validation set, then shuffled into development and test files. Therefore, the sentence-level romanization data is not to be used as the training sentence corpus. Do not treat rejoined.tsv as a training split. The aligned TSV files are derived views of the same held-out sentences, not additional independent examples.
Archive handling
- Reuse the already downloaded Hindi files if they exist in the workspace.
- If download is needed, use HTTP byte-range retrieval or a trusted file-level download. Do not scan a 1.87 GB remote tar in a way that triggers repeated multi-gigabyte reads.
- Record the URL, archive size, range-request behavior, extracted member path, extracted file size, and SHA-256 for each file.
- Do not download other language directories or the full native-script Wikipedia corpus for this project.
Known audit results to verify against the actual files
From the project owner's current Colab audit of hi.translit.sampled.train.tsv:
- Shape: 44,204 rows × 3 columns.
- Columns: native word, Romanized form, attestation count.
- 25,000 unique native words.
- 41,344 unique Romanized strings.
- Two rows have missing Romanized strings.
- No duplicate native/Romanized pairs were observed.
- Attestation count: min 1, median 2, mean about 1.92, max 10.
- 14,186 native words have more than one listed Romanization (about 56.7%).
- Distribution of unique Romanized variants per native word: 1 variant: 10,813; 2: 9,927; 3: 3,649; 4: 505; 5: 77; 6: 20; 7: 3; 8: 3; 9: 2; 0: 1.
Recompute all figures from the raw file. Do not hard-code these values. Investigate the two missing rows and the one native word with zero non-null Romanizations. Preserve the raw records and create an explicit exclusion log; do not silently delete them.
3.3 COMI-LINGUA: supplementary Hinglish reference only
Dataset: https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA
Repository: https://github.com/lingo-iitgn/COMI-LINGUA
Paper: https://aclanthology.org/2025.findings-emnlp.422/
Current dataset card declares CC BY 4.0 and lists configurations LID, MLI, POS, NER, TN, and MT.

Use COMI-LINGUA only to inspect code-mixing patterns, script composition, informal normalization phenomena, and candidate Hinglish lexical forms. It does not provide the task labels required for this project’s NLI, paraphrase, and toxicity training conditions.
What the owner has already inspected
- MLI train: 26,225 rows; annotator columns contain hi/`en` matrix-language labels. MLI is not the token-level LID configuration.
- TN train: 24,547 rows; annotations can substantially rewrite or normalize source text. Annotator disagreement is high. Do not treat the targets as interchangeable spelling variants.
- MT train: 24,692 rows; MT test: 5,000 rows; source/`Sentences` overlap between these two splits was previously found to be zero.
- MT includes original Devanagari Sentences, predicted English/Romanized Hindi/Devanagari outputs, and three annotator versions of those targets.
- The three annotator RH columns were predominantly Latin-script in the previous audit, but a small number of outputs were Devanagari or mixed-script.
- Across the three annotator RH columns, 16,475 rows had one distinct string, 6,792 had two, and 1,375 had three.
- Pairwise character similarity was very high: means about 0.9946, 0.9797, and 0.9789, with medians 1.0.
- Different strings are not necessarily different spellings of the same word. The columns can differ lexically, syntactically, or translationally.
Permitted use
- Inspect MT train outputs for supplementary code-mixing and Romanized-Hindi patterns.
- Keep MT test out of any extraction used to expand the training inventory.
- Candidate forms may be exported to a separate review file only if the extraction method preserves provenance and makes no unsupported word-alignment assumption.
- Do not automatically align source Devanagari words to RH output words by position. Translation and word-order differences make that unsafe.
- Do not turn sentence-level translation disagreements into attestation weights.
- Do not use MT outputs as NLI/paraphrase/toxicity labels.
- Do not use TN annotations as a direct Romanization variant dictionary.
- Do not merge COMI-derived candidates into the production Dakshina inventory unless they pass an explicit high-precision validation step and are marked with their source and confidence status.
The dataset card describes COMI-LINGUA as an iterative research preview with limited safety measures. Record this caveat in the source inventory.
4. Required workspace and immutable data layout
Use the existing project root if present. Otherwise create:
romcom-x/
├── data/
│   ├── raw/
│   │   ├── indi_romcom/              # exact official source files; never edit
│   │   ├── dakshina/hi/              # selected original Hindi files only
│   │   └── comi_lingua/              # selected source files; never edit
│   ├── interim/
│   │   ├── source_records/           # extracted and normalized records
│   │   ├── split_manifests/          # source-level group assignments
│   │   ├── romanization/             # lexicon, aliases, collision and coverage tables
│   │   └── qa_samples/               # deterministic review samples
│   └── processed/
│       ├── source_original/          # original task records, normalized schema
│       ├── exp01_canonical/
│       │   ├── train.jsonl
│       │   └── validation.jsonl
│       ├── exp02_realistic/
│       │   ├── train.jsonl
│       │   └── validation.jsonl
│       └── exp03_matched_noise/
│           ├── train.jsonl
│           └── validation.jsonl
├── configs/
│   └── data_augmentation.yaml
├── reports/
│   └── data_preparation/
│       ├── 00_source_access_and_license_report.md
│       ├── 01_source_schema_audit.md
│       ├── 02_dakshina_lexicon_audit.md
│       ├── 03_comi_lingua_audit.md
│       ├── 04_task_data_and_split_audit.md
│       ├── 05_romanization_inventory_report.md
│       ├── 06_augmentation_design_and_config.md
│       ├── 07_augmentation_qa_report.md
│       └── 08_final_data_handoff_report.md
├── manifests/
│   ├── source_manifest.json
│   ├── checksum_manifest.sha256
│   ├── row_provenance.jsonl
│   ├── source_split_manifest.csv
│   ├── exclusion_log.csv
│   ├── augmentation_manifest.json
│   └── processed_dataset_manifest.json
└── tests/
    ├── test_source_integrity.py
    ├── test_schema.py
    ├── test_split_isolation.py
    ├── test_variant_inventory.py
    ├── test_augmentation_invariants.py
    └── test_processed_datasets.py
Do not overwrite or normalize files under data/raw/. All transformations must write new files.
5. Data schema and provenance
5.1 Preserve task-specific structure
Use JSONL for the normalized task records. Do not flatten paired tasks or erase task-specific fields.

Required top-level fields:
{
  "example_id": "stable-record-id",
  "source_id": "original-source-group-id",
  "source_dataset": "dataset/repository-id",
  "source_revision": "commit-or-dataset-revision",
  "source_file": "original/file/path",
  "source_split": "train",
  "project_split": "train",
  "task": "nli",
  "language": "hi",
  "mixing_intensity": "25",
  "input": {},
  "target": {},
  "label": "source-label-value",
  "source_row_hash": "sha256",
  "license_ref": "source-license-record"
}
This is a required conceptual schema, not a requirement to force every task into the same nested input keys. Preserve the original fields inside input and target as needed.
- NLI must retain premise and hypothesis separately.
- Paraphrase must retain sentence 1 and sentence 2 separately.
- Toxicity must preserve the exact harmful-content field, label, and any refusal/response format required by the source task.
- Keep original labels in their source representation as well as any explicitly documented normalized label field.
- Never discard task instructions, target structure, or source annotations that may be needed later.
Use stable IDs. Prefer original source IDs. If absent, derive a deterministic hash from the canonicalized source record and record that it is derived. Keep a separate source-group ID when multiple code-mixing intensities or renderings descend from one original source prompt.
5.2 Provenance requirements
Every normalized row must be traceable to:
- Dataset ID and version/revision.
- Original file and split.
- Original row index or original ID.
- Original source ID/group ID.
- Original source-record hash.
- License record.
- Any filtering, normalization, or augmentation operation.
- Random seed and configuration for generated conditions.
No data row may appear in a processed file without a provenance record.
6. Phase A: source discovery, access, licensing, and raw acquisition
Tasks
1. Create a complete source inventory for Indi-RomCoM, Dakshina, and COMI-LINGUA.
2. Locate the canonical source and record the official URL, repository/dataset ID, exact revision, date accessed, file path, file size, checksum, and license text/URL.
3. Verify current licensing from the canonical dataset card/repository rather than relying on search snippets or old memory.
4. Record whether the license permits access, use for research, modification, redistribution of raw files, redistribution of derived files, and whether downstream model-related use is clearly addressed. Do not provide legal conclusions; mark uncertain cases for owner/legal review.
5. Keep datasets with different licenses physically and logically separated. Do not concatenate their raw data into one redistributable artifact.
6. For Dakshina, acquire only the Hindi lexicon train/dev/test files and the specific sentence-level dev/test files listed above.
7. For COMI-LINGUA, acquire only configurations needed for audit: MT train/test, TN train/test, MLI train/test, and LID only if needed to clarify the MLI/LID distinction. Do not download unrelated large assets.
8. For Indi-RomCoM, locate the actual training split; do not confuse the 8,400-instance evaluation benchmark with the described 395,076-instance fine-tuning split.
9. Create source_manifest.json and checksum_manifest.sha256.
Exit criteria
- Every used source has a pinned revision or a documented immutable release URL.
- Every raw file has a checksum.
- License status is explicit: cleared_for_internal_use, restricted, unresolved, or not_used.
- No unresolved license source is silently fed into the production training set.
- The primary task-labelled training source is identified and its actual data is accessible. If not, issue a blocker report and stop before training dataset construction.
7. Phase B: source schema and quality audits
For each source file, compute and report:
- Row count, column names, column types, and sample records.
- Missing/null/empty/whitespace-only values per field.
- Exact duplicate rows and duplicate key combinations.
- Unique counts for IDs, source texts, and labels.
- Label counts by task, language, split, and mixing intensity.
- Script composition: Latin, Devanagari, other scripts, and mixed-script proportions.
- Text length and token-count distributions.
- Malformed Unicode, replacement characters, HTML entities, control characters, and invalid encoding.
- Whether all expected files/splits are present.
- Whether paired fields have matching row counts and stable pairing.
- Whether a column name accurately describes its content (for example, RH columns that contain rare Devanagari outputs).
Use NFC normalization for derived text fields. Do not apply NFKC, aggressive transliteration normalization, lowercasing, stemming, spelling correction, or grammar correction to the canonical source. Preserve raw values in separate fields/files.
Dakshina-specific schema audit
- Verify exactly three lexicon columns: native, Romanized form, attestation count.
- Verify count is a positive integer.
- Investigate the two missing Romanized values and the zero-variant native word.
- Verify native/Romanized duplicate pairs and variant groups.
- Recompute variant-count distribution and attestation-count distribution.
- Verify that train, dev, and test are disjoint by native word, as documented by Dakshina.
- Confirm that sentence dev/test native and Romanized files have equal row counts and row-wise alignment.
- Detect duplicates within and across sentence dev/test.
- Record that rejoined.tsv and aligned files derive from the same validation-sampled sentence set; never count them as additional independent data.
COMI-LINGUA-specific audit
- Verify current configurations and actual split files.
- Distinguish MLI (matrix-language identification) from LID (token-level language identification).
- Treat TN as sentence normalization, not as orthographic-only romanization variation.
- Treat MT outputs as alternate translations/representations, not guaranteed word-by-word transliterations.
- Check source-level overlap across MT train/test and record exact results.
- Do not count distinct strings as distinct romanization variants without semantic and alignment validation.
8. Phase C: build the eligible Hindi task dataset
8.1 Filter
From the approved task-labelled source, retain only:
- Hindi (hi) training records.
- NLI / inference.
- Paraphrase detection.
- Toxicity detection.
Do not include Bengali, Gujarati, or Tamil in the training dataset. Do not include unrelated tasks from the seven-task benchmark.

Preserve all task-specific structures and labels. Never use COMI-LINGUA LID/MLI/POS/NER/TN/MT annotations as substitutes for these task labels.
8.2 Input-field role mapping
Before augmentation, create a field-role map per task:
- input_text: field(s) representing what the model must read.
- target_text: assistant output or target response.
- label: classification/relationship label.
- protected: task markers, IDs, names, numbers, math, URLs, control tokens, structural markers, and content that must remain untouched.
Only augment eligible Romanized Hindi spans in approved input fields. Never modify labels or answer schema. Do not modify targets in order to increase apparent augmentation coverage.

For each task and field, report:
- number of rows with Latin-script input;
- number with Devanagari input;
- number with mixed script;
- number with at least one unambiguous Dakshina-matched Hindi token;
- number of eligible token occurrences;
- coverage by task and split.
If the task data is not already Romanized Hindi-English code-mixed input, do not invent or automatically translate/romanize it. Report the incompatibility for owner review.
8.3 Cleaning rules
Allowed cleaning:
- Decode using the declared encoding; reject/flag undecodable content.
- Strip BOM and normalize line endings.
- Apply NFC to derived text fields.
- Normalize repeated whitespace only where it is an accidental serialization artifact; preserve meaningful newlines and task structure.
- Normalize null representation consistently while keeping source raw values.
- Remove exact duplicate rows only when they are exact duplicates of the same source record and labels agree; record all removed IDs.
Not allowed:
- Paraphrasing, translating, grammar correction, or spelling correction.
- Changing task labels.
- Removing meaningful punctuation, emoji, numbers, code snippets, or named entities.
- Merging examples because they look semantically similar.
- Removing a conflicting-label duplicate without recording and escalating it.
8.4 Duplicate and conflict handling
Run at least:
- Exact row duplicate detection.
- Exact normalized input duplicate detection within task.
- Same input with conflicting labels detection.
- Source ID duplicate/variant grouping.
- Near-duplicate candidate detection using a documented text similarity method.
Automatically remove only unambiguous exact duplicates with identical task, input, and label. Place duplicates with conflicting labels and near-duplicate candidates in review files. Do not automatically delete semantically similar but distinct examples.
9. Phase D: create leakage-safe source splits
1. Preserve official train/validation partitions if they exist and are documented.
2. If the official task source has only a training partition, create a deterministic 90/10 train/validation split at the source-group level. Use seed 42 unless the source has an official seed that must be preserved.
3. Stratify as far as possible by task, label, and original code-mixing intensity. If a stratum is too small to split, document the exception.
4. All variants/intensities of one underlying source prompt must remain in the same partition.
5. Do not split augmented rows independently.
6. Preserve source split and project split as separate fields.
7. Exact overlap between train and validation source IDs must be zero.
8. Exact overlap with reserved evaluation source IDs must be zero. If official evaluation IDs/files are accessible, use them only for overlap checking; do not inspect model outputs or use their contents to construct the augmentation inventory.
9. Run near-duplicate checks across partitions. Produce a candidate review file; automatically exclude only clear exact-source duplicates. Record every exclusion.
10. Freeze the split manifest before augmentation and hash it.
Deliver source_split_manifest.csv with at least:

source_id, example_id, task, label, language, original_split, project_split, mixing_intensity, source_hash, group_hash, exclusion_status, exclusion_reason.
10. Phase E: construct the Hindi Romanization variant inventory
10.1 Source of production variants
Use only hi.translit.sampled.train.tsv from Dakshina for the production inventory. Dakshina dev/test lexicons and sentence-level dev/test files may be used for diagnostics, but must not contribute candidate variants or counts to the training inventory.

Do not describe Dakshina attestation counts as social-media usage probabilities. They represent the number of annotator attestations for each word/romanization pair.
10.2 Normalize inventory entries
For inventory construction:
- Normalize strings to NFC.
- Use a case-folded lookup key for matching, while preserving the original surface form.
- Strip only outer whitespace.
- Do not strip internal characters, phonetic diacritics, or meaningful letter repetitions.
- Validate that native forms contain Devanagari characters and Romanized forms contain Latin letters; record exceptions.
- Require positive integer counts.
- Keep the original attestation count for each pair.
- Collapse case-only duplicates only in a separate normalized view; do not rewrite the raw lexicon.
10.3 Build native-word groups
For each native word, create:
- list of unique normalized Romanized variants;
- original surface variant forms;
- attestation count per variant;
- total attestations;
- number of variants;
- canonical candidate (unique highest-count variant, if one exists);
- canonical tie flag;
- missing/malformed status.
If multiple variants tie for the highest count, mark the word as canonical_tie=true. Do not choose an arbitrary canonical form for EXP-01; leave tied cases unchanged unless the project owner explicitly approves a tie-break policy.
10.4 Build a reverse Romanized alias index
The training source is already in Latin script, so the augmentation system needs to map an observed Latin token to a Dakshina word group.
- Create romanized_alias -> native_word_group mappings from all valid Dakshina training-lexicon variants.
- If a normalized Romanized alias maps to more than one native word, mark it as ambiguous and exclude it from automatic replacement.
- If an alias collides with a common English token, mark it as an English-collision candidate and exclude it unless trusted source token-level language tags explicitly mark it as Hindi.
- Do not use fuzzy matching, edit-distance matching, or a language model to infer the intended Hindi word in v1.
- Record every excluded ambiguous/collision alias and the native words it maps to.
- Preserve capitalization when substituting: for a sentence-initial/title-case Hindi token, capitalize the first character of the selected variant; protect all-caps forms and likely proper names.
10.5 Variant inventory outputs
Create:
- dakshina_hi_variant_inventory.tsv
- dakshina_hi_alias_index.tsv
- ambiguous_aliases.tsv
- english_collision_aliases.tsv
- canonical_ties.tsv
- invalid_lexicon_rows.tsv
Required inventory columns:

native, variant, normalized_variant, attestation_count, total_native_attestations, variant_count_for_native, canonical_candidate, is_canonical, canonical_tie, source_dataset, source_file, source_split, license_ref.
11. Phase F: coverage analysis against task inputs
This is a data coverage audit, not model evaluation.

For every input text field in every task/split:
1. Tokenize conservatively while preserving punctuation, whitespace, and original offsets.
2. Identify exact, case-insensitive matches to the high-precision Dakshina alias index.
3. Exclude ambiguous aliases, protected English collisions, named entities, all-caps abbreviations, URLs, emails, handles, hashtags, numbers, dates, mathematical expressions, code, structural markers, and task-specific protected tokens.
4. Where trusted token-level language tags exist, use them. Otherwise rely only on exact high-confidence alias matches and conservative protected-token rules.
5. Do not infer a token is Hindi merely because it resembles a Hindi word.
6. Report token-level and occurrence-weighted coverage separately.
Required metrics:
- total input tokens;
- total Latin alphabetic tokens;
- eligible unambiguous Hindi token occurrences;
- distinct eligible Hindi token types;
- token occurrence coverage over all tokens;
- coverage over Latin alphabetic tokens;
- coverage of eligible tokens that have multiple Dakshina variants;
- percentage of eligible occurrences with at least one alternative form;
- number and percentage of ambiguous aliases;
- number and percentage excluded due to English collisions;
- coverage by task, input field, original mixing intensity, and split.
Do not inflate coverage by counting the same token several times across derived augmentation conditions.

If coverage is very low for a task, report it; do not loosen matching rules silently. The owner will decide whether to change scope.
12. Phase G: augmentation generation
All three conditions must contain exactly the same source IDs, labels, task structures, and number of rows. Generate one deterministic view per source record per condition. Do not oversample one condition.
12.1 Shared transformation rules
- Apply changes only to eligible Romanized Hindi tokens in approved input fields.
- Preserve all English tokens, word order, code-switch locations, sentence boundaries, punctuation, numbers, emoji, URLs, named entities, labels, task markers, and structural tokens.
- Never add, remove, translate, or rewrite a word except for replacing one eligible Hindi token with another attested Romanized variant (EXP-02) or a matched synthetic corruption (EXP-03).
- Preserve spaces and punctuation exactly.
- Preserve task structure: premise/hypothesis, sentence pairs, toxicity prompt, etc.
- The same underlying source ID must receive deterministic output for a given condition/configuration/seed.
- Save a token-level transformation trace for every changed token.
12.2 EXP-01: canonical condition
Definition: for each eligible Romanized Hindi token, replace its form with the unique highest-attestation Dakshina training-lexicon variant for its native-word group.
- If the alias is unambiguous and the native-word group has a unique maximum-attestation form, use that form.
- If the source token is already the canonical form, leave it unchanged.
- If the native-word group has a top-count tie, leave the token unchanged and record canonical_tie.
- If the alias is ambiguous, collides with English, is protected, or is absent from the lexicon, leave it unchanged and record the reason.
- No language model or external transliterator may be used to fill gaps.
This is a canonicalized source-data condition, not a claim that the modal Dakshina spelling is the universal or socially dominant spelling.
12.3 EXP-02: realistic romanization variation
Definition: start from the canonicalized EXP-01 representation and introduce alternative attested spellings for eligible Hindi tokens.

Frozen starting policy:
- Draw one per-example aggressiveness value a from Uniform(0.10, 0.35).
- For each eligible token, replace it with probability a.
- If selected, choose a non-current alternative from the same native-word group.
- Sample alternatives proportionally to their Dakshina attestation counts, conditional on excluding the current form.
- If no alternative exists, leave unchanged.
- Do not sample from Dakshina dev/test lexicons.
- Do not invent spellings.
- Do not create a second copy or multiple augmentation views of one source record.
Attestation-weighted sampling is a reproducible proxy based on annotator attestations. It must not be described as a model of real-world user typing frequency.

Save the per-example aggressiveness value, per-token selected variant, old/new strings, attestation counts, and random seed.

The owner may later change the aggressiveness range after reviewing the prepared data. The agent must not tune it based on model results.
12.4 EXP-03: matched random-noise control
Definition: create a non-linguistic perturbation control with the same source rows and broadly matched edit magnitude as EXP-02.
- Start from the EXP-01 canonicalized input text.
- Use the exact same eligible token positions selected for change in the corresponding EXP-02 record.
- For each selected token, calculate the edit operations and character edit magnitude between the canonical spelling and the EXP-02 attested variant.
- Apply the same number and approximate mix of character substitutions, insertions, and deletions to the canonical token, but choose random character operations/letters rather than an attested variant.
- Keep the result non-empty and Latin-script; do not alter whitespace or punctuation.
- Never touch English tokens, labels, names, numbers, URLs, emoji, or task markers.
- Ensure random-noise output does not accidentally equal a known Dakshina alternative for that same native word; if collision occurs, resample deterministically and record it.
- Match per-example changed-token count and as closely as possible per-example character edit distance/length-change distribution to EXP-02.
The random-noise condition is intentionally artificial. Do not present it as natural language.
12.5 Seeds and reproducibility
- Base seed: 42.
- Derive deterministic per-example seeds from base_seed + stable_hash(example_id, condition) or another documented stable scheme.
- Do not depend on Python's process-randomized hash.
- Write the exact seed derivation into data_augmentation.yaml.
- Save generated datasets to disk; never regenerate them implicitly at training time.
- Running the pipeline twice with identical input hashes/configuration must produce byte-identical processed JSONL files and manifests, except for explicitly documented timestamps.
13. Task-specific protection rules
NLI
- Keep premise and hypothesis separate.
- Apply eligible spelling changes to either input field independently, using deterministic token seeds.
- Do not reorder the pair, alter relation labels, or change premise/hypothesis boundaries.
Paraphrase
- Keep sentence 1 and sentence 2 separate.
- Apply changes independently to eligible Hindi tokens in each input sentence.
- Do not swap the pair or change the paraphrase/non-paraphrase label.
Toxicity
- Preserve the harmful-content input exactly unless the approved source schema explicitly marks a Romanized Hindi input span as part of the input text to be augmented.
- Never soften, intensify, paraphrase, censor, or otherwise alter harmful source content.
- Never change toxicity labels or refusal/response requirements.
- Do not augment target refusal explanations in this assignment.
- If the source task's code-mixing is only in assistant outputs/targets, then input-side romanization augmentation coverage is zero for those rows. Report that clearly and flag it for owner review rather than changing the experiment silently.
14. Transformation trace and auditability
For every output record, store a parallel trace containing:
- example_id, source_id, task, split, condition;
- input field and token index/character offsets;
- original token;
- normalized lookup key;
- matched Dakshina native-word group;
- variant list and attestation counts used;
- selected replacement;
- operation type (canonicalize, attested_variant, random_noise, unchanged);
- reason for unchanged token (no_match, ambiguous_alias, english_collision, protected_token, canonical_tie, single_variant, etc.);
- aggressiveness value;
- seed;
- character edit distance and length delta;
- source-row hash and config hash.
Trace files must make every changed character/token explainable. Do not store only aggregate counts.
15. Phase H: processed dataset QA and acceptance tests
The agent must run all tests below and provide machine-readable results.
15.1 Structural invariants
For EXP-01, EXP-02, and EXP-03:
- Train row counts must match one another exactly.
- Validation row counts must match one another exactly.
- Source ID sets must match one another exactly within each split.
- Task counts and label counts must match exactly between conditions.
- All required task fields must be present.
- No unexpected null/empty input or target fields.
- All source IDs must trace to the raw source manifest.
- All row hashes must match the source provenance manifest.
15.2 Label and task invariants
- 100% label preservation.
- 100% pair/field structure preservation.
- 100% toxicity target/refusal schema preservation.
- No change to task instructions or protected markers.
- NLI premise/hypothesis identities preserved.
- Paraphrase sentence-pair order preserved.
15.3 Text protection invariants
- No English token modified unless trusted source tags explicitly mark it Hindi.
- No number/date/math token modified.
- No URL/email/handle/hashtag modified.
- No named entity or all-caps acronym modified.
- No punctuation or whitespace moved or deleted.
- No sentence boundary changed.
- No Hindi-English switch boundary changed.
Any violation is a hard failure. Do not waive it because aggregate counts look good.
15.4 Split and leakage invariants
- Exact source ID overlap between train and validation: zero.
- Exact source ID overlap between training and reserved benchmark evaluation: zero.
- Any near-duplicate candidate across splits is reported.
- All augmentations of one source group remain in one split.
- Dakshina dev/test lexicons and sentence-level dev/test data do not contribute to the training inventory or augmentation mappings.
- COMI-LINGUA MT test does not contribute to training inventory extraction.
15.5 Augmentation invariants
- EXP-01 canonical form always corresponds to a unique Dakshina training-lexicon max-count variant.
- EXP-02 every replacement is a variant attested in the Dakshina training lexicon.
- EXP-02 variant selection weights are traceable to attestation counts.
- EXP-03 changes are synthetic random character edits and are not represented as attested variants.
- EXP-03 matches EXP-02's eligible token positions and edit magnitude within a documented tolerance.
- No changed token becomes empty, whitespace-containing, or non-Latin.
- No source record receives more than one output view per condition.
- Every condition is deterministic.
15.6 Sample audit
Create a deterministic sample of at least 100 source examples, stratified across task, label, mixing intensity, and train/validation split where possible. For each source example, show:
- Original input fields.
- EXP-01 canonical fields.
- EXP-02 realistic fields.
- EXP-03 matched-noise fields.
- Token-level traces.
The agent must not claim human-validated naturalness or semantic correctness. Add review columns for a human reviewer:

meaning_preserved, English_preserved, entities_preserved, code_switch_preserved, spelling_plausible, issue_type, reviewer_comment.

Leave human-only judgment fields blank unless a real qualified reviewer completes them. The agent may annotate obvious mechanical failures, but must distinguish automated checks from human judgments.
16. Required reports
Every report must include a timestamp, code/config version, source revision(s), and links/paths to the underlying machine-readable results.
00_source_access_and_license_report.md
- Sources searched and exact URLs.
- Whether official Indi-RomCoM training data was found and accessed.
- Exact file names, revisions, checksums, and split labels.
- License and attribution summary for every source.
- Resolved/unresolved usage questions.
- Clear go/no-go decision for each source.
- Any blocker requiring owner decision.
01_source_schema_audit.md
- Schemas, row counts, nulls, duplicate counts, script statistics, field distributions, and malformed records.
- Dataset-specific findings for Indi-RomCoM, Dakshina, and COMI-LINGUA.
- Explicit source-role decision for each file.
02_dakshina_lexicon_audit.md
- All recomputed counts listed in Section 3.2.
- Missing/malformed-row investigation.
- Variant count distribution.
- Attestation count distribution.
- Train/dev/test word overlap check.
- Sample variant groups.
- Caveat that counts are annotation attestations, not social-media frequencies.
03_comi_lingua_audit.md
- Config/split schema.
- MT train/test source-overlap result.
- Script consistency results.
- Annotator distinct-string distribution and similarity summary.
- Examples showing orthographic versus lexical/translation variation.
- Strict statement that COMI-LINGUA is supplementary and not the task-label source.
04_task_data_and_split_audit.md
- Final task-source access outcome.
- Rows by task, label, language, split, and intensity.
- Field-role mapping.
- Cleaning/exclusion log summary.
- Train/validation split procedure and seed.
- Exact and near-duplicate leakage results.
- Reserved evaluation overlap result.
- Any toxicity input/target augmentation coverage caveat.
05_romanization_inventory_report.md
- Number of native words, variants, usable multi-variant words, and unique aliases.
- Ambiguous alias count and English-collision count.
- Canonical tie count.
- Missing/malformed row count.
- Example mappings and exclusions.
- Coverage on the actual task input records.
- No unsupported claim that Dakshina counts are natural typing probabilities.
06_augmentation_design_and_config.md
- Exact transformation definitions for EXP-01/02/03.
- Protected token rules.
- Mapping and sampling logic.
- Seed derivation.
- Config hash.
- Any deviations from this specification, with reasons.
07_augmentation_qa_report.md
- Processed row counts by task/split/condition.
- Label and structure preservation checks.
- Token change rates and changed characters.
- Coverage and no-op rates.
- Matched-noise edit-distance match summary.
- Protected-token violations (must be zero).
- Leakage violations (must be zero).
- Determinism test result.
- Human-review sample path and unresolved issues.
08_final_data_handoff_report.md
A concise, decision-oriented summary:
- READY_FOR_OWNER_REVIEW, BLOCKED, or NOT_READY status.
- Exact paths to the three processed dataset conditions.
- Exact paths to source, split, provenance, and checksum manifests.
- Summary of source licenses and any remaining restrictions.
- Main coverage limitations.
- Any unresolved task-specific ambiguity, especially toxicity.
- List of acceptance tests and pass/fail results.
- Specific decisions required from the project owner.
- Explicit statement: no model was loaded, evaluated, or trained.
17. Acceptance criteria for completion
The assignment is complete only when all of the following are true:
1. The primary task-labelled Hindi training source is verified and accessible, or the agent has issued a clear blocker report explaining why it cannot proceed.
2. All sources have provenance records and license status.
3. Raw files are immutable and checksummed.
4. The Hindi-only task dataset preserves all required task structure and labels.
5. Source-level train/validation split is frozen and leakage-checked.
6. Dakshina production inventory uses only its Hindi training lexicon.
7. Dakshina sentence-level dev/test files remain outside training and variant-inventory construction.
8. COMI-LINGUA remains supplementary; its MT/TN outputs are not misrepresented as word-level attested spelling variants.
9. Canonical, realistic, and matched-noise datasets exist with matching IDs, counts, labels, and split assignments.
10. Every augmentation is deterministic and traceable.
11. Protected-token, label-preservation, task-structure, and leakage tests pass.
12. A 100-example review sample is available for owner inspection.
13. All required reports and manifests exist.
14. No model training, base-model validation, model evaluation, publication, or packaging was performed.
18. Stop conditions and escalation rules
Stop and report rather than guessing if:
- Indi-RomCoM's actual task-labelled training split cannot be located or accessed.
- The license for a required task dataset is unresolved or does not clearly permit the intended internal use.
- Required source IDs or grouping information are absent and source-level split integrity cannot be established.
- The released source data is not Romanized Hindi-English code-mixed input as required by the project.
- Toxicity code-mixing exists only in targets rather than input fields, creating a mismatch with the input-robustness objective.
- More than a negligible number of English/proper-name tokens are modified by the generator.
- Any label, target structure, source ID, task field, or split assignment changes unexpectedly.
- Random-noise control cannot be matched fairly to realistic augmentation.
- Deterministic output cannot be reproduced from the same inputs and configuration.
Never fix a blocker by silently changing the research question, adding a dataset, generating synthetic labels, using held-out evaluation data, or relaxing protected-token rules.
19. Final message the agent must send to the project owner
At completion, send a short handoff containing:
1. Status: ready for owner review / blocked / not ready.
2. The primary task dataset actually used and its license status.
3. Counts for source rows and final train/validation rows by task.
4. Dakshina inventory counts and task-input coverage.
5. Exact paths to EXP-01, EXP-02, and EXP-03 files.
6. Exact paths to all reports and manifests.
7. Acceptance-test failures, unresolved issues, and decisions needed.
8. Explicit confirmation that no model was loaded, validated, evaluated, or trained.
Do not proceed into model work. The project owner will personally handle base-model evaluation, QLoRA training, model evaluation and analysis, packaging, documentation, and publication.
References
1. Das et al. (2026), Indi-RomCoM: Code-Mixed Benchmark for Evaluating LLMs on Romanized Indic-English Instructions. https://arxiv.org/abs/2606.30790
2. Roark et al. (2020), Processing South Asian Languages Written in the Latin Script: The Dakshina Dataset. https://www.aclweb.org/anthology/2020.lrec-1.294
3. Official Dakshina repository and README (dataset layout, split construction, and CC BY-SA 4.0 license). https://github.com/google-research-datasets/dakshina
4. Sheth et al. (2025), COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing. https://aclanthology.org/2025.findings-emnlp.422/
5. COMI-LINGUA Hugging Face dataset card (configurations and CC BY 4.0 metadata). https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA