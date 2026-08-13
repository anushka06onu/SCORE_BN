"""## 9. Exploratory data analysis — CPU"""

# Import required library
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Convert the text column to standard Python strings.
# This prevents the PyArrow Unicode-regex error in Google Colab.
text_python = df['text'].astype('object').fillna('').astype(str)

# Calculate text-length features
df['char_length'] = text_python.str.len()
df['word_length'] = text_python.str.split().str.len()

# Count Bangla characters using Python's regular-expression engine
df['bangla_chars'] = text_python.apply(
    lambda text: len(re.findall(r'[\u0980-\u09FF]', text))
)

# Count English/Latin characters
df['latin_chars'] = text_python.apply(
    lambda text: len(re.findall(r'[A-Za-z]', text))
)

# Calculate the proportion of Latin characters
df['latin_ratio'] = (
    df['latin_chars']
    / (
        df['bangla_chars']
        + df['latin_chars']
        + 1
    )
)

# Define the correct severity-class order
class_order = [
    'General Query',
    'Routine',
    'Urgent',
    'Emergency'
]

# Create the EDA figures
fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(18, 5)
)

# Plot 1: Class distribution
sns.countplot(
    data=df,
    x='label_name',
    order=class_order,
    ax=axes[0],
    palette='viridis'
)

axes[0].set_title('Class Distribution')
axes[0].set_xlabel('Severity Class')
axes[0].set_ylabel('Number of Queries')
axes[0].tick_params(
    axis='x',
    rotation=20
)

# Add counts above the bars
for container in axes[0].containers:
    axes[0].bar_label(
        container,
        fmt='%d',
        padding=3
    )

# Plot 2: Text-length distribution
sns.histplot(
    data=df,
    x='word_length',
    hue='label_name',
    hue_order=class_order,
    bins=40,
    element='step',
    stat='count',
    common_norm=False,
    ax=axes[1]
)

axes[1].set_title('Text Length Distribution')
axes[1].set_xlabel('Number of Words')
axes[1].set_ylabel('Number of Queries')

# Plot 3: Character length by severity class
sns.boxplot(
    data=df,
    x='label_name',
    y='char_length',
    order=class_order,
    ax=axes[2],
    palette='viridis'
)

axes[2].set_title('Character Length by Class')
axes[2].set_xlabel('Severity Class')
axes[2].set_ylabel('Number of Characters')
axes[2].tick_params(
    axis='x',
    rotation=20
)

# Improve the figure layout
plt.tight_layout()

# Create the figures folder if it does not already exist
(PROJECT_DIR / 'figures').mkdir(
    parents=True,
    exist_ok=True
)

# Save the EDA figure
plt.savefig(
    PROJECT_DIR / 'figures/basic_eda.png',
    dpi=200,
    bbox_inches='tight'
)

plt.show()

# Calculate vocabulary size
vocabulary = {
    token
    for text in text_python
    for token in text.split()
}

print('Vocabulary size:', len(vocabulary))

# Show descriptive statistics by severity class
eda_statistics = (
    df.groupby('label_name')[
        [
            'word_length',
            'char_length',
            'bangla_chars',
            'latin_chars',
            'latin_ratio'
        ]
    ]
    .describe()
    .round(2)
)

# Arrange the table in severity order
eda_statistics = eda_statistics.reindex(class_order)

display(eda_statistics)

