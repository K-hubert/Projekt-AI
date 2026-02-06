import pandas as pd
import matplotlib.pyplot as plt

# wczytanie CSV
df = pd.read_csv("tests/eval_report.csv")

# Liczenie trafności
df["accuracy"] = df["keywords_hit"] / df["keywords_total"]

# średnia trafność na metode
agg = df.groupby("retrieval_method")["accuracy"].mean()

# wykres
plt.figure()
plt.bar(agg.index, agg.values)
plt.xlabel("Metoda dopasowania dokumentów")
plt.ylabel("Średnia trafność (keywords_hit / total)")
plt.title("Porównanie metod retrieval")
plt.ylim(0, 1)

plt.show()
