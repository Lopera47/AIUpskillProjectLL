"""Industry-standard ROUGE evaluation (no HuggingFace)."""
import asyncio
import json
from pathlib import Path
from rouge_score import rouge_scorer
from litellm import completion
import os
from dotenv import load_dotenv

load_dotenv()


class IndustryRougeEvaluator:
    """REAL industry evaluation: articles → summarize → ROUGE score."""
    
    def __init__(self, test_articles_path: str = "data/evaluation/test_articles.json"):
        self.model = os.getenv("LITELLM_MODEL")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL', 'rouge2'])
        self.test_articles = self.load_test_articles(test_articles_path)
    
    def load_test_articles(self, path: str) -> list:
        """Load test articles from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def call_llm(self, prompt: str) -> str:
        """Call LLM."""
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM error: {e}")
            return ""
    
    async def summarize_article(self, article: str) -> str:
        """Summarize using your SummarizerAgent prompt."""
        prompt = f"""Summarize this article into 2-3 sentences for a daily digest.

Article:
{article}

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
    
    async def evaluate(self):
        """Run industry-standard evaluation."""
        print("=" * 70)
        print("  Industry-Standard ROUGE Evaluation")
        print("=" * 70)
        print(f"\n📊 Evaluating on {len(self.test_articles)} test articles\n")
        
        results = []
        
        for i, test_case in enumerate(self.test_articles):
            print(f"📄 Article {i+1}/{len(self.test_articles)}: ", end="")
            
            article = test_case["article"]
            reference = test_case["reference_summary"]
            
            # Your agent summarizes
            hypothesis = await self.summarize_article(article)
            print(f"✓")
            
            if not hypothesis:
                continue
            
            # Calculate ROUGE
            scores = self.calculate_rouge(reference, hypothesis)
            
            results.append({
                'id': test_case['id'],
                'article': article[:100],
                'reference': reference,
                'hypothesis': hypothesis,
                'scores': scores
            })
            
            print(f"   Reference: {reference[:60]}...")
            print(f"   Your output: {hypothesis[:60]}...")
            print(f"   ROUGE1: {scores['rouge1']:.3f}\n")
        
        # Calculate averages
        print("=" * 70)
        print("  FINAL RESULTS")
        print("=" * 70)
        
        avg_rouge1 = sum([r['scores']['rouge1'] for r in results]) / len(results)
        avg_rouge2 = sum([r['scores']['rouge2'] for r in results]) / len(results)
        avg_rougeL = sum([r['scores']['rougeL'] for r in results]) / len(results)
        
        print(f"\n📊 Your Agent (average across {len(results)} articles):")
        print(f"   ROUGE1:  {avg_rouge1:.3f}")
        print(f"   ROUGE2:  {avg_rouge2:.3f}")
        print(f"   ROUGEL:  {avg_rougeL:.3f}")
        
        print(f"\n📈 Industry Baseline:")
        print(f"   ROUGE1:  0.40-0.45")
        print(f"   ROUGE2:  0.17-0.20")
        print(f"   ROUGEL:  0.36-0.41")
        
        # Assessment
        if avg_rouge1 > 0.40:
            status = "✅ Good (at/above baseline)"
        elif avg_rouge1 > 0.30:
            status = "⚠️  Acceptable"
        else:
            status = "❌ Needs improvement"
        
        print(f"\n🎯 Status: {status}")
        
        # Save report
        await self.save_report(results, avg_rouge1, avg_rouge2, avg_rougeL)
        
        print("\n" + "=" * 70)
        print("✅ Evaluation complete!")
        print("=" * 70)
        
        return {
            'rouge1': avg_rouge1,
            'rouge2': avg_rouge2,
            'rougeL': avg_rougeL,
            'num_samples': len(results)
        }
    
    async def save_report(self, results: list, avg_r1, avg_r2, avg_rL):
        """Save report."""
        output_path = Path("data/evaluation/industry_rouge_report.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Industry-Standard ROUGE Evaluation Report\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **ROUGE1:** {avg_r1:.3f}\n")
            f.write(f"- **ROUGE2:** {avg_r2:.3f}\n")
            f.write(f"- **ROUGEL:** {avg_rL:.3f}\n")
            f.write(f"- **Samples:** {len(results)}\n\n")
            
            f.write(f"## Results\n\n")
            for r in results:
                f.write(f"### Article {r['id']}\n\n")
                f.write(f"**Reference:** {r['reference']}\n\n")
                f.write(f"**Your Output:** {r['hypothesis']}\n\n")
                f.write(f"**ROUGE1:** {r['scores']['rouge1']:.3f}\n\n")
        
        print(f"💾 Report saved to {output_path}")


async def run_evaluation():
    """Run evaluation."""
    evaluator = IndustryRougeEvaluator()
    results = await evaluator.evaluate()
    return results


if __name__ == "__main__":
    asyncio.run(run_evaluation())