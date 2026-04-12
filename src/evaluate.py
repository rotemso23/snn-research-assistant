"""
evaluate.py — Evaluation of the RAG pipeline using RAGAS.

Runs a curated set of 10 question/answer pairs and computes:
  - faithfulness:        Does the answer stay grounded in the retrieved context? (LLM-as-judge)
  - answer_relevancy:   Does the answer address the question? (LLM-as-judge)
  - context_precision:  Are the retrieved chunks actually relevant? (LLM-as-judge)
  - context_recall:     Did retrieval cover what the ground truth requires? (LLM-as-judge)
  - semantic_similarity: Cosine similarity between answer and ground truth embeddings (fast, no LLM)

Results are saved to evaluation_results.json.

Usage:
    python src/evaluate.py
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is in path and working directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall  # noqa: deprecated import path; ragas.metrics.collections requires OpenAI-only InstructorLLM
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from src.retriever import retrieve_and_rerank
from src.generator import generate

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CLAUDE_MODEL = "claude-sonnet-4-6"

# Manually curated test set — questions grounded in actual thesis content
EVAL_SET = [
    {
        "question": "What is the main goal of the thesis?",
        "ground_truth": "The thesis aims to develop energy-efficient AI models inspired by brain computation. It proposes new neural network architectures using spike-based communication and exponential/logarithmic domain computation, designed to reduce energy consumption while maintaining high performance on classification tasks.",
    },
    {
        "question": "What spiking neuron model is used in the thesis?",
        "ground_truth": "The thesis uses the Leaky Integrate-and-Fire (LIF) neuron model. In this model, the membrane potential accumulates input current over time and emits a spike when it crosses a threshold, after which it resets.",
    },
    {
        "question": "What is the key advantage of spiking neural networks over traditional ANNs?",
        "ground_truth": "Spiking neural networks are more energy-efficient because they use sparse, event-driven computation — neurons only communicate when a spike occurs. This is in contrast to traditional ANNs which perform dense matrix multiplications at every layer on every input.",
    },
    {
        "question": "What models were proposed in the thesis?",
        "ground_truth": "The thesis proposes four new AI models: the Linear Model, the Exponential Model, the Scaled Exponential Model, and the Split Scaled Exponential Model. These models are designed to be both energy-efficient and suitable for neuromorphic hardware implementation.",
    },
    {
        "question": "What datasets are used for evaluation?",
        "ground_truth": "The thesis evaluates the proposed models on EEG (electroencephalography) data for biomedical signal classification and geophone data for seismic signal classification. These are real-world datasets relevant to low-power embedded applications.",
    },
    {
        "question": "What is the role of the membrane potential in a spiking neuron?",
        "ground_truth": "The membrane potential in a spiking neuron accumulates weighted input spikes over time. When the potential exceeds a threshold, the neuron fires a spike and the potential resets. Between spikes, the potential decays (leaks) back toward a resting value.",
    },
    {
        "question": "Why is energy efficiency important for AI models?",
        "ground_truth": "Energy efficiency is critical because modern deep learning models are extremely power-hungry. Projections suggest that AI systems could consume over 9% of global electricity by 2030 if current trends continue. Neuromorphic and spike-based approaches aim to reduce this energy cost significantly.",
    },
    {
        "question": "What biological principles inspire the proposed models?",
        "ground_truth": "The proposed models draw inspiration from two fundamental properties of brain computation: spike-based communication between neurons, and exponential or logarithmic domain computation arising from ion energy following a Boltzmann distribution.",
    },
    {
        "question": "What is the Boltzmann distribution and how does it relate to the thesis?",
        "ground_truth": "The Boltzmann distribution describes the energy distribution of particles in a physical system. In the thesis, it motivates the use of exponential and logarithmic domain computations in neural models, since ion transport in biological neurons follows this distribution.",
    },
    {
        "question": "What problem does this thesis address in the context of neuromorphic computing?",
        "ground_truth": "The thesis addresses the challenge of building AI models that are both accurate and energy-efficient enough to run on neuromorphic hardware. It bridges the gap between software deep learning models and hardware-friendly spiking neural network implementations.",
    },
]


def compute_semantic_similarity(answers: list[str], ground_truths: list[str]) -> list[float]:
    model = SentenceTransformer(EMBEDDING_MODEL)
    answer_embeddings = model.encode(answers, normalize_embeddings=True)
    truth_embeddings = model.encode(ground_truths, normalize_embeddings=True)
    similarities = np.sum(answer_embeddings * truth_embeddings, axis=1)
    return similarities.tolist()


def run_evaluation() -> dict:
    """Run RAGAS evaluation and return results."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Check your .env file.")

    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(model=CLAUDE_MODEL, api_key=api_key)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    samples = []
    answers = []
    ground_truths = []
    per_question = []

    print("Generating answers for evaluation set...")
    for item in EVAL_SET:
        question = item["question"]
        ground_truth = item["ground_truth"]

        chunks = retrieve_and_rerank(question, fetch_k=20, top_k=7)
        result = generate(question, chunks)
        answer = result["answer"]
        contexts = [doc.page_content for doc in chunks]

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=ground_truth,
            )
        )
        answers.append(answer)
        ground_truths.append(ground_truth)
        per_question.append({"question": question})
        print(f"  [done] {question[:70]}...")

    print("\nComputing semantic similarity...")
    sem_sims = compute_semantic_similarity(answers, ground_truths)
    for i, sim in enumerate(sem_sims):
        per_question[i]["semantic_similarity"] = round(float(sim), 4)

    print("\nRunning RAGAS evaluation (LLM-as-judge — this takes a few minutes)...")
    dataset = EvaluationDataset(samples=samples)
    run_config = RunConfig(max_workers=2, timeout=120)
    ragas_result = evaluate(dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings, run_config=run_config)
    ragas_df = ragas_result.to_pandas()

    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for i, row in ragas_df.iterrows():
        for key in metric_keys:
            if key in row:
                per_question[i][key] = round(float(row[key]), 4) if not np.isnan(row[key]) else None

    aggregate = {
        "semantic_similarity": round(float(np.mean(sem_sims)), 4),
    }
    for key in metric_keys:
        col_values = ragas_df[key].dropna().tolist() if key in ragas_df.columns else []
        aggregate[key] = round(float(np.mean(col_values)), 4) if col_values else None

    scores = {**aggregate, "per_question": per_question}
    return scores


if __name__ == "__main__":
    scores = run_evaluation()

    output_path = "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\n{'='*50}")
    print("Evaluation results saved to", output_path)
    print(f"  faithfulness:        {scores.get('faithfulness')}")
    print(f"  answer_relevancy:    {scores.get('answer_relevancy')}")
    print(f"  context_precision:   {scores.get('context_precision')}")
    print(f"  context_recall:      {scores.get('context_recall')}")
    print(f"  semantic_similarity: {scores.get('semantic_similarity')}")
