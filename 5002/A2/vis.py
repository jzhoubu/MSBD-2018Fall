import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical(df, feats, target):
    for column in feats:
        data = df.loc[df[column].notnull(), :]
        fig = plt.figure(figsize=(18, 12))
        ax = sns.countplot(x=column, hue=target, data=data, ax = plt.subplot(211));
        plt.xlabel(column, fontsize=14);
        plt.ylabel('Number of occurrences', fontsize=14);
        plt.suptitle('Plots for '+column, fontsize=18);
        plt.legend(loc=1);
        # Adding percents over bars
        height = [p.get_height() for p in ax.patches]
        height = [x if not pd.isnull(x) else 0 for x in height]  # fix bug
        ncol = int(len(height)/2)
        total = [height[i] + height[i + ncol] for i in range(ncol)] * 2
        for i, p in enumerate(ax.patches):
            ax.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 10,
                    '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14);
        sns.pointplot(x=column, y='label', data=data, ax = plt.subplot(212));
        plt.xlabel(column, fontsize=14);
        plt.ylabel('High-income Percentage', fontsize=14);
        plt.show()
    pass;

def plot_numeric(df, feats, target):
    for column in feats:
        fig = plt.figure(figsize=(18, 12))
        sns.distplot(df[column].dropna(), ax=plt.subplot(221));
        plt.xlabel(column, fontsize=14);
        plt.ylabel('Density', fontsize=14);
        plt.suptitle('Plots for '+column, fontsize=18);
        sns.distplot(df.loc[df.label==0, column].dropna(),
                     color='red', label='low-income', ax=plt.subplot(222));
        sns.distplot(df.loc[df.label==1, column].dropna(),
                     color='blue', label='high-income', ax=plt.subplot(222));
        plt.legend(loc='best')
        plt.xlabel(column, fontsize=14);
        plt.ylabel('Density high-income / low-income', fontsize=14);

        sns.barplot(x=target, y=column, data=df, ax=plt.subplot(223));
        plt.xlabel('Income >50k or not ?', fontsize=14);
        plt.ylabel('Average ' + column, fontsize=14);

        # Box plot of Column per label / Not label Value
        sns.boxplot(x=target, y=column, data=df, ax=plt.subplot(224));
        plt.xlabel('label==1', fontsize=14);
        plt.ylabel(column, fontsize=14);
        plt.show()
    pass;