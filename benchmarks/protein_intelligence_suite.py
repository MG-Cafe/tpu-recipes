#!/usr/bin/env python3
"""
Standardized Protein Intelligence & Biological Accuracy Benchmark Suite for
GenBio AI's `genbio-ai/GB.Protein-16B` (`AIDO.Protein-16B`).

Evaluates 4 core biological intelligence tasks identically across:
  - NVIDIA G4 GPU (FP32 Reference & BF16 Best-Optimized)
  - Google Cloud TPU v7x (TorchTPU BF16 TP=2 & TP=8)
  - Google Cloud TPU v6e (TorchTPU BF16 TP=2 & TP=4, plus 1-Chip INT8 W8A16 & Pallas W8A8)
"""
import math
from typing import Dict, List, Tuple, Callable
import numpy as np

VOCAB_TOKENS = [
    "[PAD]", "L", "A", "G", "V", "S", "E", "R", "T", "I",
    "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W",
    "C", "X", "B", "U", "Z", "O", ".", "-", "[MASK]", "[gMASK]",
    "[sMASK]", "[eod]", "[sop]", "[eop]", "[SEP]", "[HC]", "[LC]", "[HUMAN]",
    "<cdr1>", "</cdr1>", "<cdr2>", "</cdr2>", "<cdr3>", "</cdr3>"
]
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB_TOKENS)}
ID_TO_TOKEN = {i: t for i, t in enumerate(VOCAB_TOKENS)}
PAD_ID = TOKEN_TO_ID["[PAD]"]    # 0
MASK_ID = TOKEN_TO_ID["[MASK]"]  # 28
SEP_ID = TOKEN_TO_ID["[SEP]"]    # 34
AA_IDS = list(range(1, 21))      # 20 standard amino acids (L..C)

# 1. Six Real Canonical UniProt Protein Sequences
PROTEIN_PANEL: Dict[str, Dict[str, str]] = {
    "GFP_P42212": {
        "name": "Aequorea victoria Green Fluorescent Protein (UniProt P42212)",
        "family": "Fluorescent Reporter",
        "seq": (
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
            "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
            "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
            "HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        ),
    },
    "HBB_P68871": {
        "name": "Homo sapiens Hemoglobin Subunit Beta (UniProt P68871)",
        "family": "Oxygen Transport Globin",
        "seq": (
            "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK"
            "VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG"
            "KEFTPPVQAAYQKVVAGVANALAHKYH"
        ),
    },
    "TP53_DBD_P04637": {
        "name": "Homo sapiens Tumor Suppressor p53 DNA-Binding Domain (UniProt P04637, aa 94-292)",
        "family": "Transcription Factor / Tumor Suppressor",
        "seq": (
            "SSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPP"
            "GTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFR"
            "HSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVR"
            "VCACPGRDRRTEEENLRKK"
        ),
    },
    "UBB_P0CG47": {
        "name": "Homo sapiens Ubiquitin (UniProt P0CG47)",
        "family": "Post-Translational Modifier",
        "seq": (
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYN"
            "IQKESTLHLVLRLRGG"
        ),
    },
    "LYZ_P61626": {
        "name": "Homo sapiens Lysozyme C (UniProt P61626)",
        "family": "Antimicrobial Hydrolase",
        "seq": (
            "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRA"
            "TNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRD"
            "PQGIRAWVAWRNRCQNRDVRQYVQGCGV"
        ),
    },
    "SARS2_RBD_ACE2": {
        "name": "SARS-CoV-2 Spike RBD (P0DTC2) + Human ACE2 Peptidase Domain (Q9BYF1) Complex",
        "family": "Viral-Host Receptor Complex",
        "seq": (
            "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFK"
            "CYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNS"
            "NNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQ"
            "PTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
            "|"
            "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQST"
            "LAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNP"
            "QECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYED"
            "YGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISP"
            "IGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSV"
            "GLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHI"
            "QYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLL"
            "KQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDP"
            "ASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRL"
            "GKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYAD"
        ),
    },
}

# 2. 24 Gold-Standard ClinVar / ProteinGym DMS Missense Variants
# Note: For TP53_DBD_P04637 (which starts at full-length residue 94), 1-based local index = full_pos - 93.
CLINVAR_DMS_VARIANTS = [
    # HBB (Hemoglobin Beta, P68871) - 4 Pathogenic Core/Heme vs 4 Benign Surface
    {"protein": "HBB_P68871", "name": "HBB_H92Y_Hb_M_Hyde_Park", "pos_1based": 93, "wt": "H", "mut": "Y", "label": "Pathogenic", "mechanism": "Proximal heme-binding histidine loss"},
    {"protein": "HBB_P68871", "name": "HBB_H63Y_Hb_M_Saskatoon", "pos_1based": 64, "wt": "H", "mut": "Y", "label": "Pathogenic", "mechanism": "Distal heme-binding histidine loss"},
    {"protein": "HBB_P68871", "name": "HBB_F42S_Hb_Hammersmith", "pos_1based": 43, "wt": "F", "mut": "S", "label": "Pathogenic", "mechanism": "Invariant heme pocket Phe disruption"},
    {"protein": "HBB_P68871", "name": "HBB_L28P_Hb_Genova", "pos_1based": 29, "wt": "L", "mut": "P", "label": "Pathogenic", "mechanism": "B-helix proline helix-breaker"},
    {"protein": "HBB_P68871", "name": "HBB_E6A_Hb_G_Makassar", "pos_1based": 7, "wt": "E", "mut": "A", "label": "Benign", "mechanism": "Asymptomatic surface polymorphism"},
    {"protein": "HBB_P68871", "name": "HBB_G16D_Hb_J_Baltimore", "pos_1based": 17, "wt": "G", "mut": "D", "label": "Benign", "mechanism": "Benign surface substitution"},
    {"protein": "HBB_P68871", "name": "HBB_K59N_Hb_J_Kaohsiung", "pos_1based": 60, "wt": "K", "mut": "N", "label": "Benign", "mechanism": "Benign solvent-exposed Lys->Asn"},
    {"protein": "HBB_P68871", "name": "HBB_T87Q_Surface_Neutral", "pos_1based": 88, "wt": "T", "mut": "Q", "label": "Benign", "mechanism": "Neutral F-helix surface residue"},

    # TP53 DNA-Binding Domain (P04637, offset=93) - 4 Pathogenic Cancer Hotspots vs 4 Benign/Neutral
    {"protein": "TP53_DBD_P04637", "name": "TP53_R175H_Zinc_Hotspot", "pos_1based": 175 - 93, "wt": "R", "mut": "H", "label": "Pathogenic", "mechanism": "L2 loop zinc-binding structural hotspot"},
    {"protein": "TP53_DBD_P04637", "name": "TP53_C176F_Zinc_Coord", "pos_1based": 176 - 93, "wt": "C", "mut": "F", "label": "Pathogenic", "mechanism": "Direct Zn2+ coordinating cysteine loss"},
    {"protein": "TP53_DBD_P04637", "name": "TP53_G245S_L3_Hotspot", "pos_1based": 245 - 93, "wt": "G", "mut": "S", "label": "Pathogenic", "mechanism": "Minor groove L3 loop conformational hotspot"},
    {"protein": "TP53_DBD_P04637", "name": "TP53_R282W_Helix_Hotspot", "pos_1based": 282 - 93, "wt": "R", "mut": "W", "label": "Pathogenic", "mechanism": "H2 helix packing & DNA binding hotspot"},
    {"protein": "TP53_DBD_P04637", "name": "TP53_V217M_ClinVar_Benign", "pos_1based": 217 - 93, "wt": "V", "mut": "M", "label": "Benign", "mechanism": "ClinVar benign conservative hydrophobic swap"},
    {"protein": "TP53_DBD_P04637", "name": "TP53_S96A_Nterm_Neutral", "pos_1based": 96 - 93, "wt": "S", "mut": "A", "label": "Benign", "mechanism": "Unstructured N-terminal flank neutral"},
    {"protein": "TP53_DBD_P04637", "name": "TP53_K291R_Cterm_Neutral", "pos_1based": 291 - 93, "wt": "K", "mut": "R", "label": "Benign", "mechanism": "Conservative positively charged C-tail swap"},
    {"protein": "TP53_DBD_P04637", "name": "TP53_E224D_Loop_Neutral", "pos_1based": 224 - 93, "wt": "E", "mut": "D", "label": "Benign", "mechanism": "Solvent-exposed acidic loop conservative swap"},

    # GFP (P42212) - 4 Loss-of-Fluorescence Core/Chromophore vs 4 Stabilizing/Neutral Variants
    {"protein": "GFP_P42212", "name": "GFP_G67A_Chromophore_Dead", "pos_1based": 67, "wt": "G", "mut": "A", "label": "Pathogenic", "mechanism": "Abolishes autocatalytic chromophore cyclization"},
    {"protein": "GFP_P42212", "name": "GFP_W57G_Core_Collapse", "pos_1based": 57, "wt": "W", "mut": "G", "label": "Pathogenic", "mechanism": "Destabilizes buried central helix Trp"},
    {"protein": "GFP_P42212", "name": "GFP_F27A_BetaBarrel_Core", "pos_1based": 27, "wt": "F", "mut": "A", "label": "Pathogenic", "mechanism": "Creates cavity in beta-barrel hydrophobic core"},
    {"protein": "GFP_P42212", "name": "GFP_R96A_Catalytic_Arg", "pos_1based": 96, "wt": "R", "mut": "A", "label": "Pathogenic", "mechanism": "Removes catalytic Arg96 required for maturation"},
    {"protein": "GFP_P42212", "name": "GFP_C48A_CysFree_Stable", "pos_1based": 48, "wt": "C", "mut": "A", "label": "Benign", "mechanism": "Prevents surface disulfide oligomerization"},
    {"protein": "GFP_P42212", "name": "GFP_V163A_Cycle3_Folder", "pos_1based": 163, "wt": "V", "mut": "A", "label": "Benign", "mechanism": "Well-known Cycle-3 folding enhancer"},
    {"protein": "GFP_P42212", "name": "GFP_F99S_Surface_Soluble", "pos_1based": 99, "wt": "F", "mut": "S", "label": "Benign", "mechanism": "Increases solubility on beta-barrel surface"},
    {"protein": "GFP_P42212", "name": "GFP_I167T_Conservative_Neutral", "pos_1based": 167, "wt": "I", "mut": "T", "label": "Benign", "mechanism": "Tolerated thermostable variant"},
]


def encode_protein_sequence(seq: str, pad_to_multiple: int = 256) -> Tuple[np.ndarray, int]:
    """Encodes a single or multi-chain ('|'-separated) protein sequence with trailing [SEP]."""
    ids: List[int] = []
    for ch in seq:
        if ch == "|":
            ids.append(SEP_ID)
        else:
            ids.append(TOKEN_TO_ID[ch])
    ids.append(SEP_ID)
    valid_len = len(ids)
    padded_len = ((valid_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    arr = np.full((1, padded_len), PAD_ID, dtype=np.int32)
    arr[0, :valid_len] = ids
    return arr, valid_len


def log_softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits_f32 = logits.astype(np.float64)
    m = np.max(logits_f32, axis=axis, keepdims=True)
    shifted = logits_f32 - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_softmax_np(logits, axis=axis))


def compute_auroc(scores_positive: List[float], scores_negative: List[float]) -> float:
    """Computes exact Mann-Whitney U / Wilcoxon AUROC where Benign > Pathogenic."""
    pairs = 0
    wins = 0.0
    for sp in scores_positive:
        for sn in scores_negative:
            pairs += 1
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return wins / max(pairs, 1)


def evaluate_intelligence_suite(
    forward_fn: Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray]],
    chip_name: str,
) -> Dict:
    """
    Executes the 4-task Protein Intelligence & Biological Benchmark Suite.
    `forward_fn(input_ids_np, valid_len)` must return `(logits_np, hidden_states_np)`
    where `logits_np` has shape `[1, padded_len, 44]` and `hidden_states_np` has shape `[1, padded_len, 2304]`.
    """
    results: Dict = {
        "chip_name": chip_name,
        "mlm_per_protein": {},
        "clinvar_dms_variants": {},
        "embeddings": {},
        "raw_logits_sample": {},
    }

    total_masked_tokens = 0
    total_nll = 0.0
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0
    total_entropy = 0.0

    embedding_vectors = []
    embedding_labels = []
    embedding_families = []

    # Task 1 & Task 3: MLM Perplexity, Masked Amino Acid Recovery & Sequence Embeddings
    for key, info in PROTEIN_PANEL.items():
        seq = info["seq"]
        clean_ids, valid_len = encode_protein_sequence(seq, pad_to_multiple=256)

        # Unmasked pass for sequence embedding & unmasked logit reference
        logits_unmasked, hidden_unmasked = forward_fn(clean_ids, valid_len)
        seq_emb = hidden_unmasked[0, :valid_len - 1, :].astype(np.float64).mean(axis=0)
        seq_emb_norm = seq_emb / (np.linalg.norm(seq_emb) + 1e-12)
        embedding_vectors.append(seq_emb_norm)
        embedding_labels.append(key)
        embedding_families.append(info["family"])

        if key == "GFP_P42212":
            results["raw_logits_sample"]["GFP_P42212_unmasked"] = logits_unmasked[0, :valid_len, :21].astype(np.float32)
            results["raw_logits_sample"]["GFP_P42212_hidden"] = seq_emb_norm.astype(np.float32)

        # Deterministic 15% masking pass (every 7th standard amino acid position: idx % 7 == 3)
        masked_ids = clean_ids.copy()
        masked_positions = []
        true_token_ids = []
        for pos in range(valid_len - 1):
            tid = int(clean_ids[0, pos])
            if tid in AA_IDS and (pos % 7 == 3):
                masked_ids[0, pos] = MASK_ID
                masked_positions.append(pos)
                true_token_ids.append(tid)

        logits_masked, _ = forward_fn(masked_ids, valid_len)
        # Restrict log_softmax over the 20 standard amino acids (token IDs 1..20)
        aa_logits = logits_masked[0, masked_positions, 1:21]
        aa_log_probs = log_softmax_np(aa_logits, axis=-1)
        aa_probs = np.exp(aa_log_probs)

        prot_nll = 0.0
        prot_top1 = 0
        prot_top3 = 0
        prot_top5 = 0
        prot_entropy = 0.0

        for idx_m, true_tid in enumerate(true_token_ids):
            aa_idx = true_tid - 1  # 0..19
            lp = float(aa_log_probs[idx_m, aa_idx])
            prot_nll += -lp
            ranked = np.argsort(-aa_logits[idx_m])
            if ranked[0] == aa_idx:
                prot_top1 += 1
            if aa_idx in ranked[:3]:
                prot_top3 += 1
            if aa_idx in ranked[:5]:
                prot_top5 += 1
            ent = -float(np.sum(aa_probs[idx_m] * aa_log_probs[idx_m]))
            prot_entropy += ent

        n_m = len(masked_positions)
        total_masked_tokens += n_m
        total_nll += prot_nll
        total_top1 += prot_top1
        total_top3 += prot_top3
        total_top5 += prot_top5
        total_entropy += prot_entropy

        results["mlm_per_protein"][key] = {
            "length_tokens": valid_len,
            "num_masked": n_m,
            "mean_nll": round(prot_nll / n_m, 5),
            "perplexity": round(math.exp(prot_nll / n_m), 4),
            "top1_accuracy": round(prot_top1 / n_m, 5),
            "top3_accuracy": round(prot_top3 / n_m, 5),
            "top5_accuracy": round(prot_top5 / n_m, 5),
            "mean_entropy_nats": round(prot_entropy / n_m, 5),
        }

    results["mlm_aggregate"] = {
        "total_masked_residues": total_masked_tokens,
        "mean_nll": round(total_nll / total_masked_tokens, 5),
        "perplexity": round(math.exp(total_nll / total_masked_tokens), 4),
        "top1_accuracy": round(total_top1 / total_masked_tokens, 5),
        "top3_accuracy": round(total_top3 / total_masked_tokens, 5),
        "top5_accuracy": round(total_top5 / total_masked_tokens, 5),
        "mean_entropy_nats": round(total_entropy / total_masked_tokens, 5),
    }

    # Task 2: Zero-Shot ClinVar / DMS Missense Variant Effect Prediction (24 Gold-Standard Variants)
    # Group variants by protein so we can evaluate masked marginals cleanly
    by_protein: Dict[str, List[Dict]] = {}
    for v in CLINVAR_DMS_VARIANTS:
        by_protein.setdefault(v["protein"], []).append(v)

    benign_scores = []
    pathogenic_scores = []

    for prot_key, vlist in by_protein.items():
        seq = PROTEIN_PANEL[prot_key]["seq"]
        clean_ids, valid_len = encode_protein_sequence(seq, pad_to_multiple=256)

        for v in vlist:
            pos0 = v["pos_1based"] - 1
            wt_aa = seq[pos0]
            assert wt_aa == v["wt"], f"Mismatch at {v['name']}: expected {v['wt']}, got {wt_aa}"
            masked_single = clean_ids.copy()
            masked_single[0, pos0] = MASK_ID
            logits_m, hidden_m = forward_fn(masked_single, valid_len)
            log_probs_pos = log_softmax_np(logits_m[0, pos0, 1:21], axis=-1)
            wt_idx = TOKEN_TO_ID[v["wt"]] - 1
            mut_idx = TOKEN_TO_ID[v["mut"]] - 1
            delta_ll = float(log_probs_pos[mut_idx] - log_probs_pos[wt_idx])

            # Also compute mutant sequence embedding for homology clustering
            mut_ids = clean_ids.copy()
            mut_ids[0, pos0] = TOKEN_TO_ID[v["mut"]]
            _, hidden_mut = forward_fn(mut_ids, valid_len)
            mut_emb = hidden_mut[0, :valid_len - 1, :].astype(np.float64).mean(axis=0)
            mut_emb_norm = mut_emb / (np.linalg.norm(mut_emb) + 1e-12)
            embedding_vectors.append(mut_emb_norm)
            embedding_labels.append(v["name"])
            embedding_families.append(PROTEIN_PANEL[prot_key]["family"])

            results["clinvar_dms_variants"][v["name"]] = {
                "protein": prot_key,
                "pos_1based": v["pos_1based"],
                "wt": v["wt"],
                "mut": v["mut"],
                "label": v["label"],
                "delta_log_prob": round(delta_ll, 5),
                "wt_log_prob": round(float(log_probs_pos[wt_idx]), 5),
                "mut_log_prob": round(float(log_probs_pos[mut_idx]), 5),
            }
            if v["label"] == "Benign":
                benign_scores.append(delta_ll)
            else:
                pathogenic_scores.append(delta_ll)

    auroc = compute_auroc(benign_scores, pathogenic_scores)
    # Optimal threshold separating Benign (>= threshold) from Pathogenic (< threshold)
    all_thresholds = sorted(benign_scores + pathogenic_scores)
    best_acc = 0.0
    best_thresh = -1.5
    for th in all_thresholds:
        correct = sum(1 for s in benign_scores if s >= th) + sum(1 for s in pathogenic_scores if s < th)
        acc = correct / (len(benign_scores) + len(pathogenic_scores))
        if acc > best_acc:
            best_acc = acc
            best_thresh = th

    results["clinvar_dms_summary"] = {
        "num_variants": len(benign_scores) + len(pathogenic_scores),
        "num_benign": len(benign_scores),
        "num_pathogenic": len(pathogenic_scores),
        "mean_benign_delta_log_prob": round(float(np.mean(benign_scores)), 4),
        "mean_pathogenic_delta_log_prob": round(float(np.mean(pathogenic_scores)), 4),
        "separation_gap_log_prob": round(float(np.mean(benign_scores) - np.mean(pathogenic_scores)), 4),
        "auroc": round(auroc, 5),
        "classification_accuracy": round(best_acc, 5),
        "optimal_threshold": round(best_thresh, 4),
    }

    # Task 3: Sequence Embedding Similarity Matrix & Family Homology Retrieval Recall@1
    emb_mat = np.stack(embedding_vectors, axis=0)  # [N_seq, 2304]
    sim_matrix = emb_mat @ emb_mat.T               # [N_seq, N_seq]
    # For each variant/sequence that has a same-family partner, check if top-1 nearest neighbor is same family
    retrievals = 0
    correct_top1 = 0
    for i in range(len(embedding_labels)):
        fam_i = embedding_families[i]
        same_fam_count = sum(1 for j in range(len(embedding_labels)) if j != i and embedding_families[j] == fam_i)
        if same_fam_count > 0:
            retrievals += 1
            sims = sim_matrix[i].copy()
            sims[i] = -999.0
            nn_idx = int(np.argmax(sims))
            if embedding_families[nn_idx] == fam_i:
                correct_top1 += 1

    results["embedding_summary"] = {
        "num_sequences_embedded": len(embedding_labels),
        "embedding_dim": int(emb_mat.shape[1]),
        "family_retrieval_recall_at_1": round(correct_top1 / max(retrievals, 1), 5),
    }
    results["raw_logits_sample"]["pairwise_sim_matrix"] = sim_matrix.astype(np.float32)

    return results


def compare_against_reference(test_res: Dict, ref_res: Dict) -> Dict:
    """Computes cross-chip intelligence fidelity metrics against the GPU reference."""
    v_names = sorted(ref_res["clinvar_dms_variants"].keys())
    ref_dms = np.array([ref_res["clinvar_dms_variants"][k]["delta_log_prob"] for k in v_names], dtype=np.float64)
    tst_dms = np.array([test_res["clinvar_dms_variants"][k]["delta_log_prob"] for k in v_names], dtype=np.float64)

    pearson_r = float(np.corrcoef(ref_dms, tst_dms)[0, 1])
    # Spearman rank correlation
    ref_ranks = np.argsort(np.argsort(ref_dms)).astype(np.float64)
    tst_ranks = np.argsort(np.argsort(tst_dms)).astype(np.float64)
    spearman_rho = float(np.corrcoef(ref_ranks, tst_ranks)[0, 1])
    dms_mae = float(np.mean(np.abs(ref_dms - tst_dms)))

    # Probability distribution KL and JS divergence on GFP_P42212 unmasked logits
    ref_logits = np.asarray(ref_res["raw_logits_sample"]["GFP_P42212_unmasked"], dtype=np.float64)
    tst_logits = np.asarray(test_res["raw_logits_sample"]["GFP_P42212_unmasked"], dtype=np.float64)
    p = softmax_np(ref_logits, axis=-1)
    q = softmax_np(tst_logits, axis=-1)
    m = 0.5 * (p + q)
    kl_pq = float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=-1)))
    js_div = float(0.5 * np.mean(
        np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12)), axis=-1) +
        np.sum(q * (np.log(q + 1e-12) - np.log(m + 1e-12)), axis=-1)
    ))
    logit_cos = float(
        np.sum(ref_logits * tst_logits) /
        (np.linalg.norm(ref_logits) * np.linalg.norm(tst_logits) + 1e-12)
    )
    ref_emb = np.asarray(ref_res["raw_logits_sample"]["GFP_P42212_hidden"], dtype=np.float64)
    tst_emb = np.asarray(test_res["raw_logits_sample"]["GFP_P42212_hidden"], dtype=np.float64)
    emb_cos = float(np.sum(ref_emb * tst_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(tst_emb) + 1e-12))

    ref_sim = np.asarray(ref_res["raw_logits_sample"]["pairwise_sim_matrix"], dtype=np.float64).flatten()
    tst_sim = np.asarray(test_res["raw_logits_sample"]["pairwise_sim_matrix"], dtype=np.float64).flatten()
    sim_matrix_corr = float(np.corrcoef(ref_sim, tst_sim)[0, 1])

    return {
        "dms_pearson_r": round(pearson_r, 6),
        "dms_spearman_rho": round(spearman_rho, 6),
        "dms_mae_log_prob": round(dms_mae, 5),
        "mlm_perplexity_delta": round(test_res["mlm_aggregate"]["perplexity"] - ref_res["mlm_aggregate"]["perplexity"], 4),
        "mlm_top1_acc_delta": round(test_res["mlm_aggregate"]["top1_accuracy"] - ref_res["mlm_aggregate"]["top1_accuracy"], 5),
        "mlm_top5_acc_delta": round(test_res["mlm_aggregate"]["top5_accuracy"] - ref_res["mlm_aggregate"]["top5_accuracy"], 5),
        "clinvar_auroc_delta": round(test_res["clinvar_dms_summary"]["auroc"] - ref_res["clinvar_dms_summary"]["auroc"], 5),
        "logit_cosine_similarity": round(logit_cos, 6),
        "embedding_cosine_similarity": round(emb_cos, 6),
        "pairwise_similarity_matrix_r": round(sim_matrix_corr, 6),
        "kl_divergence_nats": round(kl_pq, 6),
        "js_divergence_nats": round(js_div, 6),
    }
