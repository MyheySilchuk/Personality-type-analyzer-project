import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def categorical_aspects(data_frame: pd.DataFrame):


    # Group by personality and each feature separately
    fear = data_frame.groupby(
    [data_frame['personality'].apply(lambda x: x.value), data_frame['stage_fear']]
    ).size().unstack(fill_value=0)
    
    drained = data_frame.groupby(
    [data_frame['personality'].apply(lambda x: x.value), data_frame['drained_after_socializing']]
    ).size().unstack(fill_value=0)
    # Create a bar chart with two sets of bars for each personality type

    labels = fear.index
    x = np.arange(len(labels))  
    width = 0.1
    

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create bars for each category (Stage Fear and Drained) for both personality types
    bars1 = ax.bar(x - 0.5 * width, fear[1], width, label='stage fear: yes', color='#e74c3c')
    bars2 = ax.bar(x - 0.5 * width, fear[0], width, fear[1], label='stage fear: no', color="#2ecc71")
    bars3 = ax.bar(x + 0.5 * width, drained[1], width, label='drained: yes', color='#3498db')
    bars4 = ax.bar(x + 0.5 * width, drained[0], width, drained[1], label='drained: no', color="#f39c12")
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    # Labels and title  
    ax.set_xlabel('personality type')
    ax.set_ylabel('number of people')
    ax.set_title('stage fear & social drain by personality type')
    ax.set_xticks(x)

    ax.set_xticklabels(sorted(data_frame['personality'].apply(lambda x: x.name).unique(), reverse=True))
    ax.set_ylim(0, max(fear.max().max(), drained.max().max()) * 1.2)

    # Add labels to the bars
    for bars in [bars1, bars2, bars3, bars4]:
        ax.bar_label(bars, label_type='center')

    # Show the plot
    plt.tight_layout()
    plt.show()




def social_behavior(data_frame):
    social_behavior_columns = ['social_event_attendance', 'going_outside', 'post_frequency']
    colors = ['#87CEEB', '#90EE90', '#FFB6C1']

    # Data for introverts and extroverts
    introverts_data = [data_frame[data_frame['personality'].apply(lambda x: x.value) == 0][col].dropna().values for col in social_behavior_columns]
    extroverts_data = [data_frame[data_frame['personality'].apply(lambda x: x.value) == 1][col].dropna().values for col in social_behavior_columns]
    
    # Combine data for both personality types
    all_data = [introverts_data, extroverts_data]

    #Create a horizontal boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Positions for the y-axis
    y_positions=[1, 2]

    # Distance between boxes
    offset = 0.3

    for t in range(len(y_positions)): 
        for i, data in enumerate(all_data[t]):
            pos = y_positions[t] - offset + i * offset
            bplot = ax.boxplot(data, positions=[pos], vert=False, patch_artist=True, 
                               widths=0.15, notch=True, showfliers=False, medianprops={'color': 'black'})
            for patch in bplot['boxes']:
                patch.set_facecolor(colors[i])
            median = np.median(data)
            ax.text(median, pos, social_behavior_columns[i], ha='center', va='center', fontsize=8, 
                    fontweight='bold', color='black')

   
    # Set y-ticks and labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted(data_frame['personality'].apply(lambda x: x.name).unique(), reverse=True))

    # Set x-axis label and title
    ax.set_xlabel('times per week')
    ax.set_title('social behavior by personality type')

    plt.tight_layout()
    plt.show()


def social_status(data_frame: pd.DataFrame):

    data_for_x_labels = ['friends_circle_size', 'time_spent_alone']

    # Create a figure with two subplots
    plt.figure(figsize=(10, 5))

    # Loop through each personality type and plot the data
    for personality_type_number in range(len(data_frame['personality'].unique())):
        plt.subplot(1, 2, personality_type_number+1)
        sns.histplot(
            data=data_frame,
            x=data_for_x_labels[personality_type_number],
            hue=data_frame['personality'].apply(lambda x: x.name),
            kde=True,
            palette="Set1",
            element='step',
        )

        # Set the title and labels for each subplot
        plt.title(f' {data_for_x_labels[personality_type_number]} : introverts VS extraverts', fontsize=12)
        plt.xlabel(f'{data_for_x_labels[personality_type_number]}', fontsize=10)
        plt.ylabel('number of people', fontsize=12)


    plt.tight_layout()
    plt.show()
