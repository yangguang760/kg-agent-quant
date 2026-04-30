"""
KG-AgentQuant CLI

Command-line interface for KG-AgentQuant.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg_quant import KGFeatureGenerator, KGExplainer
from kg_quant.utils import setup_logger


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="KG-AgentQuant: Knowledge Graph Enhanced Alpha Factor Research"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Factor generation command
    gen_parser = subparsers.add_parser('generate', help='Generate alpha factors')
    gen_parser.add_argument('--kg-dir', default='data/kg', help='Knowledge graph directory')
    gen_parser.add_argument('--factor-json', default='data/sample/factors_sample.json',
                          help='Factor JSON file')
    gen_parser.add_argument('--type', default='quality', choices=['value', 'quality', 'momentum', 'size'],
                          help='Factor type')
    gen_parser.add_argument('--n-features', type=int, default=10, help='Number of features')
    gen_parser.add_argument('--output', help='Output file path')

    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Explain factor expressions')
    explain_parser.add_argument('--kg-dir', default='data/kg', help='Knowledge graph directory')
    explain_parser.add_argument('expressions', nargs='+', help='Factor expressions to explain')

    # KG stats command
    stats_parser = subparsers.add_parser('stats', help='Show knowledge graph statistics')
    stats_parser.add_argument('--kg-dir', default='data/kg', help='Knowledge graph directory')

    args = parser.parse_args()

    if args.command == 'generate':
        print(f"Generating {args.n_features} {args.type} factors...")
        generator = KGFeatureGenerator(kg_dir=args.kg_dir, factor_json_path=args.factor_json)
        features = generator.generate_kg_features(factor_type=args.type, n_features=args.n_features)
        print(f"Generated {features.shape[1]} features")
        if args.output:
            features.to_parquet(args.output)
            print(f"Saved to {args.output}")
        else:
            print(features.head())

    elif args.command == 'explain':
        explainer = KGExplainer(kg_dir=args.kg_dir)
        for expr in args.expressions:
            print(f"\n{'=' * 60}")
            print(f"Factor: {expr}")
            print('=' * 60)
            explanation = explainer.explain_factor(expr)
            print(f"Name: {explanation.factor_name}")
            print(f"Confidence: {explanation.explanation_confidence:.2f}")
            print(f"Indicators: {', '.join(explanation.used_indicators)}")
            print(f"Theories: {', '.join(explanation.used_theories)}")
            print(f"Patterns: {', '.join(explanation.used_patterns)}")
            print(f"\nLogic: {explanation.economic_logic}")

    elif args.command == 'stats':
        generator = KGFeatureGenerator(kg_dir=args.kg_dir)
        stats = generator.retriever.get_statistics()
        print("Knowledge Graph Statistics:")
        print("-" * 40)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()