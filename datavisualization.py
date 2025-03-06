import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Summary Statistics file
file_path = '/Users/cronebanan/Documents/DTU/Notes/S25/02450 Introduction to Machine Learning and Data Mining/Projects/MachineLearning/Data4_final.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

attributes = df.columns.to_numpy()

fig, axs = plt.subplots(15,8, figsize=(20,20))
axs = axs.flatten()

for i, attribute in enumerate(attributes):
    axs[i].boxplot(df[attribute])
    axs[i].set_title(attribute, fontsize=5)
    axs[i].set_xticks([])

for j in range(i + 1, len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.show()


columnsData = ["byg026Opførelsesår", "byg038SamletBygningsareal", "byg404Koordinat_easting", "byg404Koordinat_northing", "virkningFra_unix"]

# Histograms to plot (NUMBER, AREA, TIMESTAMP, CODELIST)
selectedAtt = ["byg026Opførelsesår", "byg038SamletBygningsareal", "registreringFra_unix"]

corr_matrix = df.corr(numeric_only=True)

"""
heatmap_segment_size = 15

num_columns = len(corr_matrix.columns)
for i in range(0, num_columns, heatmap_segment_size):
    for j in range(0, num_columns, heatmap_segment_size):
        subset_corr_matrix = corr_matrix.iloc[i:i+heatmap_segment_size, j:j+heatmap_segment_size]

        plt.figure(figsize=(10, 8))
        sns.heatmap(subset_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap - Columns {} to {}".format(i+1, i+heatmap_segment_size))
        plt.show()
"""


# Heatmap seaborn
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.05)
plt.xticks(ticks=[], labels=[])
plt.yticks(ticks=[], labels=[])
plt.title('Heatmap of all attributes')
plt.show()

# Histograms in same plot
fig, axs = plt.subplots(3)

axs[0].hist(df[columnsData[0]], bins=100, edgecolor='black')
axs[0].set_title('Distribution of Year of Construction')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Frequency')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

axs[1].hist(df[selectedAtt[1]], bins=150, edgecolor='black')
axs[1].set_title('Distribution of Total Building Area')
axs[1].set_xlabel('Area')
axs[1].set_ylabel('Frequency')
axs[1].grid(axis='y', linestyle='--', alpha=0.7)


axs[2].hist(df[selectedAtt[2]], bins=150, edgecolor='black')
axs[2].set_title('Distribution of Time since Registration')
axs[2].set_xlabel('Time (Unix)')
axs[2].set_ylabel('Frequency')
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

"""

# Histogram of 'Opførelsesår' NUMBER
plt.hist(df[columnsData[0]], bins=100, edgecolor='black')
plt.title('Distribution of Year of Construction')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Histogram of 'Samlet bygningsareal' AREA
plt.hist(df[selectedAtt[1]], bins=150, edgecolor='black')
plt.title('Distribution of Total Building Area')
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Histogram of 'Registrering fra' TIMESTAMP
plt.hist(df[selectedAtt[2]], bins=150, edgecolor='black')
plt.title('Distribution of Time since Registration')
plt.xlabel('Time (Unix)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Boxplot of 'Opførelsesår'
plt.boxplot(df[columnsData[0]])
plt.title('Boxplot of Year of Construction')
plt.show()

# Without outliers/only main distribution

# Q1 and Q3
Q1 = df[columnsData[0]].quantile(0.25)
Q3 = df[columnsData[0]].quantile(0.75)

# IQR
IQR = Q3 - Q1

# Upper and lower bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_filtered = df[(df[columnsData[0]] >= lower_bound) & 
                 (df[columnsData[0]] <= upper_bound)]

plt.figure(figsize=(6, 4))
plt.boxplot(df_filtered[columnsData[0]], vert=True, patch_artist=True)
plt.title("Year of Construction (Without Outliers)")
plt.ylabel("Year")
plt.show()



# Boxplot of 'Samlet Bygningsareal'
plt.boxplot(df[columnsData[1]], patch_artist=True)
plt.title('Boxplot of Total Building Area')
plt.xlabel('Total Building Area')
plt.show()

# Without outliers/only main distribution

# Q1 and Q3
Q1 = df[columnsData[1]].quantile(0.25)
Q3 = df[columnsData[1]].quantile(0.75)

# IQR
IQR = Q3 - Q1

# Upper and lower bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_filtered = df[(df[columnsData[1]] >= lower_bound) & 
                 (df[columnsData[1]] <= upper_bound)]

plt.figure(figsize=(6, 4))
plt.boxplot(df_filtered[columnsData[1]], vert=True, patch_artist=True)
plt.title("Total Building Area (Without Outliers)")
plt.ylabel("Area")
plt.show()


# Histogram of 'Easting'
plt.hist(df[columnsData[2]] / 1e6, bins=30, edgecolor='black')
plt.title('Distribution of Easting coordinates')
plt.locator_params(axis="x", nbins=8)
plt.xlabel('Easting Coordinate (mil)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Histogram of 'Northing'
plt.hist(df[columnsData[3]] / 1e6, bins=30, edgecolor='black')
plt.title('Distribution of Northing coordinates')
plt.locator_params(axis="x", nbins=8)
plt.xlabel('Northing Coordinate (mil)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Scatter plot matrix
num_attributes = len(columnsData)

# Grid of subplots
fig, axes = plt.subplots(num_attributes, num_attributes, figsize=(12,12))

# Plot each attribute against the others

for i in range (num_attributes):
    for j in range (num_attributes):
        ax = axes[i,j]

        ax.scatter(df[columnsData[j]], df[columnsData[i]], alpha=0.5, s=5)
        
        if j == 0:
            ax.set_ylabel(columnsData[i])
        if i == num_attributes - 1:
            ax.set_xlabel(columnsData[j])
            

plt.tight_layout()
plt.show()


# Scatter plot of 'Easting vs Northing'

color_map = {
    0: 'blue',
    1: 'yellow'
}

df['color'] = df['kommunekode_101'].map(color_map)
plt.figure(figsize=(10,6))
scatter = plt.scatter(df[columnsData[2]], df[columnsData[3]], c=df['color'], edgecolors='black')

plt.xlabel('Easting')
plt.ylabel('Northing')
plt.title('Scatter plot of Easting vs Northing colored by Kommunekode_101')

plt.scatter([], [], c='blue', label='Lyngby-Taarbæk Kommune', edgecolors='black')
plt.scatter([], [], c='yellow', label='Københavns Kommune', edgecolors='black')

plt.legend(title="Kommunekode_101")

plt.show()

# Heatmap
# Kommentarer til alle plots
# Tjek outliers (Boxplot til hver attribut)


plt.boxplot(df[columnsData[1]], patch_artist=True)
plt.title('Boxplot of Total Building Area')
plt.xlabel('Total Building Area')
plt.show()
"""
