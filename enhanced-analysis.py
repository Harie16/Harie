# Step 8: Enhanced Visualization Functions - replace the existing create_visualizations function
def create_visualizations(df, output_dir='./visualizations'):
    """Create enhanced visualizations from the analyzed data."""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set style
    plt.style.use('ggplot')
    
    # 1. Rating Distribution
    if 'rating' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='rating', data=df, palette='viridis')
        plt.title('Distribution of Ratings', fontsize=16)
        plt.xlabel('Rating', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f'{output_dir}/rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment_category'].value_counts()
    ax = sns.countplot(x='sentiment_category', data=df, palette='RdYlGn', order=['Positive', 'Neutral', 'Negative'])
    plt.title('Sentiment Distribution', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add percentage labels on top of bars
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12)
    
    plt.savefig(f'{output_dir}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Emotion Distribution
    emotion_columns = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral']
    emotion_counts = df['dominant_emotion'].value_counts()
    
    plt.figure(figsize=(12, 7))
    bars = emotion_counts.plot(kind='bar', color=sns.color_palette('viridis', len(emotion_counts)))
    plt.title('Dominant Emotions in Reviews', fontsize=16)
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add percentage labels
    for i, v in enumerate(emotion_counts):
        plt.text(i, v + 0.5, f'{100 * v / total:.1f}%', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/emotion_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Aspect Sentiment
    aspects = ['performance', 'battery', 'display', 'design', 'keyboard', 'price', 'software', 'support', 'ports', 'audio', 'cooling']
    aspect_sentiments = []
    aspect_counts = []
    
    for aspect in aspects:
        if f'has_{aspect}' in df.columns and f'{aspect}_sentiment' in df.columns:
            count = df[f'has_{aspect}'].sum()
            mean_sentiment = df[df[f'has_{aspect}']][f'{aspect}_sentiment'].mean()
            if not np.isnan(mean_sentiment) and count > 0:
                aspect_sentiments.append((aspect, mean_sentiment, count))
    
    if aspect_sentiments:
        # Sort by frequency of mention
        aspect_sentiments.sort(key=lambda x: x[2], reverse=True)
        aspects, sentiments, counts = zip(*aspect_sentiments)
        
        # Create figure
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x=list(aspects), y=list(sentiments), palette='RdYlGn_r')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Average Sentiment by Aspect', fontsize=16)
        plt.xlabel('Aspect (with mention count)', fontsize=14)
        plt.ylabel('Average Sentiment (-1 to 1)', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add sentiment values and count on top of bars
        for i, (v, c) in enumerate(zip(sentiments, counts)):
            ax.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}\n({c})', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/aspect_sentiment.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Intent Distribution
    plt.figure(figsize=(12, 7))
    intent_order = df['primary_intent'].value_counts().index
    ax = sns.countplot(y='primary_intent', data=df, order=intent_order, palette='Set2')
    plt.title('Customer Intent Distribution', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Intent Category', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        percentage = f'{100 * p.get_width() / total:.1f}%'
        ax.annotate(percentage, (p.get_width(), p.get_y() + p.get_height()/2),
                   xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/intent_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Word Cloud of positive and negative reviews
    try:
        from wordcloud import WordCloud
        
        # Positive word cloud
        positive_text = ' '.join(df[df['sentiment_category'] == 'Positive']['full_text'])
        if positive_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                max_words=100, contour_width=3, contour_color='steelblue').generate(positive_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud - Positive Reviews', fontsize=16)
            plt.savefig(f'{output_dir}/positive_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Negative word cloud
        negative_text = ' '.join(df[df['sentiment_category'] == 'Negative']['full_text'])
        if negative_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                max_words=100, contour_width=3, contour_color='firebrick').generate(negative_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud - Negative Reviews', fontsize=16)
            plt.savefig(f'{output_dir}/negative_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
    except ImportError:
        print("WordCloud not available. Skipping word cloud visualizations.")
    
    # NEW VISUALIZATION 1: Most Common Issues in Reviews
    plt.figure(figsize=(14, 8))
    issue_columns = [col for col in df.columns if col.startswith('issue_')]
    
    if issue_columns:
        issue_counts = df[issue_columns].sum().sort_values(ascending=False)
        issue_names = [col.replace('issue_', '').replace('_', ' ') for col in issue_counts.index]
        
        # Plot top 10 issues
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x=issue_counts.values[:10], y=issue_names[:10], palette='Reds_r')
        plt.title('Top 10 Issues Mentioned in Reviews', fontsize=16)
        plt.xlabel('Count', fontsize=14)
        plt.ylabel('Issue', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add percentage labels
        for i, v in enumerate(issue_counts.values[:10]):
            percentage = f'{100 * v / total:.1f}%'
            ax.text(v + 0.5, i, percentage, va='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/common_issues.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # NEW VISUALIZATION 2: Most Common Words in Positive vs Negative Reviews
    try:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from collections import Counter
        
        plt.figure(figsize=(16, 10))
        
        # Process positive reviews
        positive_reviews = df[df['sentiment_category'] == 'Positive']
        positive_words = []
        
        for text in positive_reviews['full_text']:
            words = word_tokenize(str(text).lower())
            words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]
            positive_words.extend(words)
            
        # Process negative reviews  
        negative_reviews = df[df['sentiment_category'] == 'Negative']
        negative_words = []
        
        for text in negative_reviews['full_text']:
            words = word_tokenize(str(text).lower())
            words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]
            negative_words.extend(words)
        
        # Get top words
        positive_counter = Counter(positive_words).most_common(15)
        negative_counter = Counter(negative_words).most_common(15)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot positive words
        pos_words, pos_counts = zip(*positive_counter) if positive_counter else ([], [])
        ax1.barh(pos_words, pos_counts, color='green')
        ax1.set_title('Most Common Words in Positive Reviews', fontsize=14)
        ax1.set_xlabel('Count', fontsize=12)
        
        # Plot negative words
        neg_words, neg_counts = zip(*negative_counter) if negative_counter else ([], [])
        ax2.barh(neg_words, neg_counts, color='red')
        ax2.set_title('Most Common Words in Negative Reviews', fontsize=14)
        ax2.set_xlabel('Count', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/positive_negative_words.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    except ImportError:
        print("NLTK not available. Skipping word comparison visualization.")
    
    # NEW VISUALIZATION 3: User Satisfaction by Product
    if 'product' in df.columns and 'rating' in df.columns:
        plt.figure(figsize=(14, 8))
        product_ratings = df.groupby('product')['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        product_ratings = product_ratings[product_ratings['count'] >= 3]  # Only products with at least 3 reviews
        
        if not product_ratings.empty:
            ax = sns.barplot(x=product_ratings.index, y=product_ratings['mean'], palette='Blues_d')
            plt.title('Average User Satisfaction by Product', fontsize=16)
            plt.xlabel('Product', fontsize=14)
            plt.ylabel('Average Rating', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=12)
            
            # Add count labels
            for i, (_, row) in enumerate(product_ratings.iterrows()):
                ax.text(i, row['mean'] - 0.3, f'n={int(row["count"])}', ha='center', fontsize=10, color='white')
            
            plt.ylim(0, 5.5)  # Assuming 5-star rating scale
            plt.tight_layout()
            plt.savefig(f'{output_dir}/product_satisfaction.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # NEW VISUALIZATION 4: Sentiment Comparison Across Aspects
    if aspect_sentiments:
        # Create separate dataframes for positive and negative sentiment scores
        aspect_data = []
        
        for aspect in aspects:
            if f'has_{aspect}' in df.columns and f'{aspect}_sentiment' in df.columns:
                # Get positive sentiment scores
                pos_sentiment = df[(df[f'has_{aspect}']) & (df[f'{aspect}_sentiment'] > 0)][f'{aspect}_sentiment'].mean()
                pos_count = len(df[(df[f'has_{aspect}']) & (df[f'{aspect}_sentiment'] > 0)])
                
                # Get negative sentiment scores
                neg_sentiment = df[(df[f'has_{aspect}']) & (df[f'{aspect}_sentiment'] < 0)][f'{aspect}_sentiment'].mean()
                neg_count = len(df[(df[f'has_{aspect}']) & (df[f'{aspect}_sentiment'] < 0)])
                
                if not np.isnan(pos_sentiment) or not np.isnan(neg_sentiment):
                    aspect_data.append({
                        'aspect': aspect,
                        'positive': pos_sentiment if not np.isnan(pos_sentiment) else 0,
                        'negative': abs(neg_sentiment) if not np.isnan(neg_sentiment) else 0,
                        'pos_count': pos_count,
                        'neg_count': neg_count
                    })
        
        if aspect_data:
            aspect_df = pd.DataFrame(aspect_data)
            aspect_df = aspect_df.sort_values('positive', ascending=False)
            
            # Create stacked bar chart
            plt.figure(figsize=(14, 8))
            
            # Create positions for bars
            pos = np.arange(len(aspect_df))
            width = 0.35
            
            # Create bars
            plt.bar(pos, aspect_df['positive'], width, color='green', alpha=0.7, label='Positive Sentiment')
            plt.bar(pos, aspect_df['negative'], width, bottom=aspect_df['positive'], color='red', alpha=0.7, label='Negative Sentiment')
            
            # Add count annotations
            for i, row in enumerate(aspect_df.itertuples()):
                plt.text(i, row.positive/2, f'n={row.pos_count}', ha='center', va='center', color='white', fontweight='bold')
                plt.text(i, row.positive + row.negative/2, f'n={row.neg_count}', ha='center', va='center', color='white', fontweight='bold')
            
            plt.xlabel('Aspect', fontsize=14)
            plt.ylabel('Sentiment Strength', fontsize=14)
            plt.title('Positive vs Negative Sentiment Strength by Aspect', fontsize=16)
            plt.xticks(pos, aspect_df['aspect'], rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(f'{output_dir}/aspect_sentiment_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # NEW VISUALIZATION 5: Correlation Matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    relevant_cols = [col for col in numeric_cols if 
                    'sentiment' in col or 
                    'rating' in col or 
                    col in ['textblob_polarity', 'textblob_subjectivity'] or
                    col in ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust']]
    
    if len(relevant_cols) > 1:
        plt.figure(figsize=(14, 12))
        correlation_matrix = df[relevant_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, 
                   vmin=-1, vmax=1, square=True, linewidths=.5)
        plt.title('Correlation Matrix of Sentiment Features', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

# Step 12: Generate Comprehensive Report
def generate_report(df, stats, output_dir='./'):
    """Generate a comprehensive analysis report."""
    import os
    from datetime import datetime
    
    report_path = os.path.join(output_dir, 'laptop_sentiment_analysis_report.md')
    
    # Calculate additional metrics for the report
    total_reviews = stats.get('total_reviews', 0)
    avg_rating = stats.get('avg_rating', 'N/A')
    
    # Calculate sentiment percentages
    sentiment_counts = df['sentiment_category'].value_counts()
    sentiment_pct = (sentiment_counts / len(df) * 100).round(1).to_dict()
    
    # Get top mentioned aspects
    aspect_mentions = {}
    for aspect in ['performance', 'battery', 'display', 'design', 'keyboard', 'price', 'software', 'support', 'ports', 'audio', 'cooling']:
        if f'has_{aspect}' in df.columns:  
            mentions = df[f'has_{aspect}'].sum()
            if mentions > 0:
                aspect_mentions[aspect] = mentions
    
    sorted_aspects = sorted(aspect_mentions.items(), key=lambda x: x[1], reverse=True)
    
    # Get top issues
    issue_columns = [col for col in df.columns if col.startswith('issue_')]
    issue_counts = {}
    for col in issue_columns:
        issue_name = col.replace('issue_', '').replace('_', ' ')
        issue_counts[issue_name] = df[col].sum()
    
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Generate report markdown
    with open(report_path, 'w') as f:
        f.write("# Laptop Sentiment Analysis Report\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Overview section
        f.write("## Overview\n\n")
        f.write(f"- **Total Reviews Analyzed**: {total_reviews}\n")
        if avg_rating != 'N/A':
            f.write(f"- **Average Rating**: {avg_rating:.2f} / 5.0\n")
        
        # Sentiment Distribution
        f.write("\n## Sentiment Distribution\n\n")
        f.write("| Sentiment | Count | Percentage |\n")
        f.write("|-----------|-------|------------|\n")
        for sentiment, count in sentiment_counts.items():
            f.write(f"| {sentiment} | {count} | {sentiment_pct.get(sentiment, 0)}% |\n")
        
        # Key Aspects
        f.write("\n## Key Aspects Mentioned\n\n")
        f.write("| Aspect | Mentions | % of Reviews |\n")
        f.write("|--------|----------|-------------|\n")
        for aspect, count in sorted_aspects[:8]:  # Top 8 aspects
            f.write(f"| {aspect.capitalize()} | {count} | {count/total_reviews*100:.1f}% |\n")
        
        # Top Issues
        f.write("\n## Top Issues Mentioned\n\n")
        f.write("| Issue | Mentions | % of Reviews |\n")
        f.write("|-------|----------|-------------|\n")
        for issue, count in sorted_issues[:8]:  # Top 8 issues
            f.write(f"| {issue.capitalize()} | {count} | {count/total_reviews*100:.1f}% |\n")
        
        # Aspect Sentiment Analysis
        f.write("\n## Aspect Sentiment Analysis\n\n")
        f.write("| Aspect | Sentiment Score | Positive Mentions | Negative Mentions |\n")
        f.write("|--------|----------------|-------------------|-------------------|\n")
        
        for aspect in [a for a, _ in sorted_aspects]:
            if f'{aspect}_sentiment' in df.columns:
                avg_sentiment = df[df[f'has_{aspect}']][f'{aspect}_sentiment'].mean()
                if not np.isnan(avg_sentiment):
                    pos_count = len(df[(df[f'has_{aspect}']) & (df[f'{aspect}_sentiment'] > 0)])
                    neg_count = len(df[(df[f'has_{aspect}']) & (df[f'{aspect}_sentiment'] < 0)])
                    f.write(f"| {aspect.capitalize()} | {avg_sentiment:.3f} | {pos_count} | {neg_count} |\n")
        
        # Key Insights
        f.write("\n## Key Insights\n\n")
        
        # Calculate some insights
        if 'rating' in df.columns:
            rating_distribution = df['rating'].value_counts(normalize=True).sort_index() * 100
            high_ratings = rating_distribution.get(4, 0) + rating_distribution.get(5, 0)
            low_ratings = rating_distribution.get(1, 0) + rating_distribution.get(2, 0)
            
            f.write(f"- {high_ratings:.1f}% of reviews gave 4-5 stars, while {low_ratings:.1f}% gave 1-2 stars.\n")
        
        # Most positive aspects
        positive_aspects = []
        for aspect in [a for a, _ in sorted_aspects]:
            if f'{aspect}_sentiment' in df.columns:
                sentiment = df[df[f'has_{aspect}']][f'{aspect}_sentiment'].mean()
                if not pd.isna(sentiment) and sentiment > 0.2:
                    positive_aspects.append((aspect, sentiment))
        
        if positive_aspects:
            positive_aspects.sort(key=lambda x: x[1], reverse=True)
            pos_aspect_str = ', '.join([f"{aspect}" for aspect, _ in positive_aspects[:3]])
            f.write(f"- Most positively mentioned aspects: {pos_aspect_str}.\n")
        
        # Most negative aspects
        negative_aspects = []
        for aspect in [a for a, _ in sorted_aspects]:
            if f'{aspect}_sentiment' in df.columns:
                sentiment = df[df[f'has_{aspect}']][f'{aspect}_sentiment'].mean()
                if not pd.isna(sentiment) and sentiment < -0.2:
                    negative_aspects.append((aspect, sentiment))
        
        if negative_aspects:
            negative_aspects.sort(key=lambda x: x[1])
            neg_aspect_str = ', '.join([f"{aspect}" for aspect, _ in negative_aspects[:3]])
            f.write(f"- Most negatively mentioned aspects: {neg_aspect_str}.\n")
        
        # Most common emotions
        if 'dominant_emotion' in df.columns:
            top_emotions = df['dominant_emotion'].value_counts().head(2).index.tolist()
            f.write(f"- Most common emotions expressed: {', '.join(top_emotions)}.\n")
        
        # Common words in positive reviews
        if hasattr(df, 'attrs') and 'common_positive_words' in df.attrs:
            positive_words = [word for word, _ in df.attrs['common_positive_words'][:5]]
            f.write(f"- Common words in positive reviews: {', '.join(positive_words)}.\n")
        
        # Common words in negative reviews
        if hasattr(df, 'attrs') and 'common_negative_words' in df.attrs:
            negative_words = [word for word, _ in df.attrs['common_negative_words'][:5]]
            f.write(f"- Common words in negative reviews: {', '.join(negative_words)}.\n")
        
        # Recommendations section
        f.write("\n## Recommendations\n\n")
        
        # Generate some recommendations based on the data
        if negative_aspects:
            f.write("### Areas for Improvement\n\n")
            for aspect, sentiment in negative_aspects[:3]:
                f.write(f"- Consider addressing issues related to **{aspect}** as this received consistently negative feedback.\n")
        
        if positive_aspects:
            f.write("\n### Strengths to Emphasize\n\n")
            for aspect, sentiment in positive_aspects[:3]:
                f.write(f"- Continue to highlight **{aspect}** in marketing materials as this is viewed positively by users.\n")
        
        # Add figures
        f.write("\n## Visualization Gallery\n\n")
        
        visualization_files = [
            'sentiment_distribution.png',
            'aspect_sentiment.png',
            'common_issues.png',
            'positive_negative_words.png',
            'product_satisfaction.png',
            'aspect_sentiment_comparison.png'
        ]
        
        for viz_file in visualization_files:
            if os.path.exists(os.path.join(output_dir, viz_file)):
                f.write(f"### {viz_file.replace('_', ' ').replace('.png', '').title()}\n\n")
                f.write(f"![{viz_file.replace('.png', '')}]({viz_file})\n\n")
        
        # Purchase likelihood prediction
        f.write("\n## Purchase Likelihood Prediction\n\n")
        f.write("Based on sentiment analysis and aspect evaluation, we can predict the likelihood of new users purchasing this product:\n\n")
        
        # Calculate a simple purchase likelihood score
        positive_sentiment_pct = sentiment_pct.get('Positive', 0)
        neutral_sentiment_pct = sentiment_pct.get('Neutral', 0)
        
        if avg_rating != 'N/A':
            purchase_score = (avg_rating / 5 * 0.5) + (positive_sentiment_pct / 100 * 0.3) + (neutral_sentiment_pct / 100 * 0.1)
            purchase_likelihood = "High" if purchase_score > 0.7 else "Medium" if purchase_score > 0.5 else "Low"
            
            f.write(f"- **Purchase Likelihood Score**: {purchase_score:.2f}\n")
            f.write(f"- **Likelihood Category**: {purchase_likelihood}\n\n")
            
            if purchase_likelihood == "High":
                f.write("New users are likely to purchase this product based on the high ratings and positive sentiment expressed in reviews.\n")
            elif purchase_likelihood == "Medium":
                f.write("New users may consider purchasing this product, but would benefit from addressing the key concerns identified in this analysis.\n")
            else:
                f.write("New users are less likely to purchase this product without significant improvements in the negatively reviewed aspects.\n")
        
        # End note
        f.write("\n---\n")
        f.write("*This report was automatically generated based on sentiment analysis of customer reviews.*\n")
    
    print(f"Report generated and saved to {report_path}")
    return report_path

# Step 13: Add Purchase Likelihood Prediction Model
def predict_purchase_likelihood(df):
    """Build a model to predict purchase likelihood for new users."""
    print("Building purchase likelihood prediction model...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import classification_report, accuracy_score
        
        # Create purchase indicator based on rating and sentiment
        if 'rating' in df.columns:
            # Define positive purchase indicator (e.g., rating >= 4)
            df['would_purchase'] = df['rating'] >= 4
        else: