def plot_score_comparisons(consistencies, scores, columns, max_possible_score, existing_score):
    scores = np.array(scores).reshape(len(consistencies), len(columns))
    data = pd.DataFrame(data=scores, index=consistencies, columns=col)
    fig, ax = plt.subplots()
    data.plot(ax=ax)
    ax.set(xlabel='Consistency', ylabel='Accuracy')
    ax.set_title('Score Comparison', fontsize=14, fontweight='bold')
    plt.show()
    # fig.savefig('score_comparison.png', transparent=False, dpi=80, inches='tight')