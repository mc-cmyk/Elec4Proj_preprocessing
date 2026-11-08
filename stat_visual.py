import seaborn as sns
import matplotlib.pyplot as plt

dataset = sns.load_dataset('tips')

print("=" * 50)
print(f"Rows: {dataset.shape[0]} Columns: {dataset.shape[1]}")
print("=" * 50)
print(dataset.head())
print(dataset.tail())


#tip distribution
sns.boxplot(
    data = dataset,
    x = 'day',
    y = 'tip',
    hue = 'sex'
)

plt.title("Tip per Day, Separated by sex")
plt.xlabel("Day of the week")
plt.ylabel("Tips")
plt.show()


#tip concentration
sns.scatterplot(
    data = dataset,
    x = 'tip',
    y = 'total_bill',
    hue= 'day'
)
plt.title("Tip concentraation")
plt.ylabel("Bill Paid")
plt.xlabel("Tip")
plt.show()

plt.stackplot('month', 'tip', 'day')
plt.show()