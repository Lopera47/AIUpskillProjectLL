"""CNN/DailyMail ROUGE evaluation for Summarizer."""
import asyncio
import os
from datasets import load_dataset
from rouge_score import rouge_scorer
from pathlib import Path
from litellm import completion
from dotenv import load_dotenv  # Add this import

# Load .env file
load_dotenv()

# Get token from env
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ['HF_TOKEN'] = hf_token


class CNNRougeEvaluator:
    """Evaluate summarizer on CNN/DailyMail with ROUGE metric."""
    
    def __init__(self):
        self.model = os.getenv("LITELLM_MODEL")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL', 'rouge2'])
    
    def call_llm(self, prompt: str) -> str:
        """Call LLM to summarize."""
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM error: {e}")
            return ""
    
    async def summarize_article(self, article_text: str) -> str:
        """Summarize article using SAME prompt as SummarizerAgent."""
        
        # Adapted from your SummarizerAgent._summarize_topic()
        prompt = f"""Summarize this article into 2-3 sentences for a daily digest.

        Article:
        {article_text[:1000]}

        Focus on main themes and key developments. Be concise and informative.

        Summary:"""
        
        summary = self.call_llm(prompt)
        return summary.strip()
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> dict:
        """Calculate ROUGE scores."""
        scores = self.rouge_scorer.score(reference, hypothesis)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    async def evaluate_cnn_dataset(self, num_samples: int = 50):
        """Evaluate on CNN/DailyMail dataset."""
        print("=" * 70)
        print("  CNN/DailyMail ROUGE Evaluation")
        print("=" * 70)
        
        # Load dataset
        print(f"\n📥 Loading CNN/DailyMail dataset...")
        try:
            #dataset = load_dataset("cnn_dailymail/cnn_dailymail", "3.0.0", split="validation")
            dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="validation", trust_remote_code=True)
            #dataset = load_dataset("GEM/xsum", split="validation")
            print(f"✅ Dataset loaded")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
        
        print(f"📊 Evaluating on {num_samples} articles...\n")
        
        results = []
        
        # Process articles
        for i, example in enumerate(dataset.take(num_samples)):
            # Progress indicator
            if i % 10 == 0 and i > 0:
                avg_rouge1 = sum([r['scores']['rouge1'] for r in results]) / len(results)
                print(f"   [{i}/{num_samples}] Current avg ROUGE1: {avg_rouge1:.3f}")
            
            article = example["article"]
            reference = example["highlights"]
            
            # Your agent summarizes
            try:
                hypothesis = await self.summarize_article(article)
            except Exception as e:
                print(f"   ❌ Error on article {i}: {e}")
                continue
            
            if not hypothesis:
                continue
            
            # Calculate ROUGE
            scores = self.calculate_rouge(reference, hypothesis)
            
            results.append({
                'id': i,
                'article_preview': article[:100],
                'reference': reference,
                'hypothesis': hypothesis,
                'scores': scores
            })
            
            # Print first few samples
            if i < 3:
                print(f"\n📄 Sample {i+1}:")
                print(f"   Article: {article[:80]}...")
                print(f"   Reference: {reference[:80]}...")
                print(f"   Your output: {hypothesis[:80]}...")
                print(f"   ROUGE1: {scores['rouge1']:.3f} | ROUGE2: {scores['rouge2']:.3f} | ROUGEL: {scores['rougeL']:.3f}")
        
        # Calculate averages
        print("\n" + "=" * 70)
        print("  FINAL RESULTS")
        print("=" * 70)
        
        if not results:
            print("❌ No results to evaluate")
            return None
        
        avg_rouge1 = sum([r['scores']['rouge1'] for r in results]) / len(results)
        avg_rouge2 = sum([r['scores']['rouge2'] for r in results]) / len(results)
        avg_rougeL = sum([r['scores']['rougeL'] for r in results]) / len(results)
        
        print(f"\n📊 Your Agent Metrics (average across {len(results)} articles):")
        print(f"   ROUGE1:  {avg_rouge1:.3f}")
        print(f"   ROUGE2:  {avg_rouge2:.3f}")
        print(f"   ROUGEL:  {avg_rougeL:.3f}")
        
        print(f"\n📈 Industry Baseline (CNN/DailyMail):")
        print(f"   ROUGE1:  0.40-0.45")
        print(f"   ROUGE2:  0.17-0.20")
        print(f"   ROUGEL:  0.36-0.41")
        
        print(f"\n🎯 Your Agent Performance:")
        if avg_rouge1 > 0.45:
            status = "✅✅ Excellent (above baseline)"
        elif avg_rouge1 > 0.40:
            status = "✅ Good (at/above baseline)"
        elif avg_rouge1 > 0.30:
            status = "⚠️  Acceptable"
        else:
            status = "❌ Needs improvement"
        
        print(f"   {status}")
        
        # Save detailed report
        await self.save_report(results, avg_rouge1, avg_rouge2, avg_rougeL)
        
        print("\n" + "=" * 70)
        print("✅ Evaluation complete!")
        print("=" * 70)
        
        return {
            'rouge1': avg_rouge1,
            'rouge2': avg_rouge2,
            'rougeL': avg_rougeL,
            'num_samples': len(results),
            'status': status
        }
    
    async def save_report(self, results: list, avg_r1: float, avg_r2: float, avg_rL: float):
        """Save detailed report to markdown."""
        output_path = Path("data/evaluation/cnn_rouge_report.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# CNN/DailyMail ROUGE Evaluation Report\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **ROUGE1 Average:** {avg_r1:.3f}\n")
            f.write(f"- **ROUGE2 Average:** {avg_r2:.3f}\n")
            f.write(f"- **ROUGEL Average:** {avg_rL:.3f}\n")
            f.write(f"- **Total Articles:** {len(results)}\n\n")
            
            f.write("## Baseline Comparison\n\n")
            f.write(f"| Metric | Your Agent | Baseline | Status |\n")
            f.write(f"|--------|-----------|----------|--------|\n")
            f.write(f"| ROUGE1 | {avg_r1:.3f} | 0.40-0.45 | {'✅' if avg_r1 > 0.40 else '❌'} |\n")
            f.write(f"| ROUGE2 | {avg_r2:.3f} | 0.17-0.20 | {'✅' if avg_r2 > 0.17 else '❌'} |\n")
            f.write(f"| ROUGEL | {avg_rL:.3f} | 0.36-0.41 | {'✅' if avg_rL > 0.36 else '❌'} |\n\n")
            
            f.write("## Sample Results\n\n")
            
            for result in results[:5]:  # Show first 5
                f.write(f"### Article {result['id'] + 1}\n\n")
                f.write(f"**Article Preview:** {result['article_preview']}...\n\n")
                f.write(f"**Reference Summary:** {result['reference']}\n\n")
                f.write(f"**Your Agent Output:** {result['hypothesis']}\n\n")
                f.write(f"**Scores:**\n")
                f.write(f"- ROUGE1: {result['scores']['rouge1']:.3f}\n")
                f.write(f"- ROUGE2: {result['scores']['rouge2']:.3f}\n")
                f.write(f"- ROUGEL: {result['scores']['rougeL']:.3f}\n\n")
                f.write("---\n\n")
        
        print(f"💾 Report saved to {output_path}")


async def run_cnn_evaluation():
    """Run CNN/DailyMail evaluation."""
    evaluator = CNNRougeEvaluator()
    results = await evaluator.evaluate_cnn_dataset(num_samples=50)
    return results


if __name__ == "__main__":
    asyncio.run(run_cnn_evaluation())