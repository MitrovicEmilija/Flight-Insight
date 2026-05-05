"""
FlightInsight — Great Expectations checkpoint runner.

Pipeline ga pokliče iz dvc.yaml stage 'validate':
  cd gx && uv run python run_checkpoint.py

Validira data/preprocessed/flights.csv proti flights_suite,
ustvari HTML poročilo v gx/uncommitted/data_docs/.

Exit code:
  0 - uspeh ALI predhodno obstoječ checkpoint failed (drift je pričakovan)
  1 - kritična napaka (suite ne obstaja, podatki manjkajo)
"""

import sys

import great_expectations as gx


def main():
    print("=" * 60)
    print("Great Expectations validation")
    print("=" * 60)

    # Pridobi context (poženemo iz mape gx/)
    context = gx.get_context()

    # Preveri da checkpoint obstaja
    checkpoint_name = "flights_checkpoint"
    try:
        checkpoint = context.get_checkpoint(checkpoint_name)
    except Exception as e:
        print(f"NAPAKA: Checkpoint '{checkpoint_name}' ne obstaja!")
        print("Najprej zaženi gx/create_suite.ipynb da ustvariš suite + checkpoint.")
        print(f"Detail: {e}")
        sys.exit(1)

    # Poženi validacijo
    print(f"Running checkpoint: {checkpoint_name}")
    result = checkpoint.run(run_id="flights_validation_run")

    # Ustvari HTML docs
    context.build_data_docs()

    # Izpiši rezultat
    if result["success"]:
        print("\n✅ All expectations passed!")
    else:
        print("\n⚠️  Some expectations failed (this is OK for monitoring pipelines).")

    # Statistika
    stats = result.get("run_results", {})
    for run_id, run_result in stats.items():
        validation_result = run_result.get("validation_result", {})
        statistics = validation_result.get("statistics", {})
        print(f"\nStatistika:")
        print(f"  Evaluated:  {statistics.get('evaluated_expectations', 0)}")
        print(f"  Successful: {statistics.get('successful_expectations', 0)}")
        print(f"  Unsuccessful: {statistics.get('unsuccessful_expectations', 0)}")
        success_pct = statistics.get('success_percent', 0)
        print(f"  Success %:  {success_pct:.1f}%")

    print(f"\nData Docs: gx/uncommitted/data_docs/local_site/index.html")

    # Vrni 0 — pipeline gre naprej tudi če je validacija failed,
    # ker je drift pričakovan v monitoring pipelines
    sys.exit(0)


if __name__ == "__main__":
    main()