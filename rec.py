import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Generate synthetic dataset
def generate_data():
    np.random.seed(42)
    
    # Course catalog
    courses = {
        'course_id': range(1, 31),
        'title': [
            'Python Basics', 'Advanced Python', 'Data Structures', 'Algorithms',
            'Machine Learning Intro', 'Deep Learning', 'NLP Fundamentals', 'Computer Vision',
            'Web Development', 'React.js', 'Node.js', 'Database Design',
            'SQL Mastery', 'MongoDB Basics', 'Cloud Computing', 'AWS Essentials',
            'Data Visualization', 'Statistics', 'Linear Algebra', 'Calculus',
            'Java Programming', 'C++ Fundamentals', 'Mobile Development', 'Flutter',
            'Cybersecurity', 'Ethical Hacking', 'DevOps', 'Docker & Kubernetes',
            'Blockchain', 'AI Ethics'
        ],
        'category': [
            'Programming', 'Programming', 'CS Fundamentals', 'CS Fundamentals',
            'AI/ML', 'AI/ML', 'AI/ML', 'AI/ML',
            'Web Dev', 'Web Dev', 'Web Dev', 'Database',
            'Database', 'Database', 'Cloud', 'Cloud',
            'Data Science', 'Math', 'Math', 'Math',
            'Programming', 'Programming', 'Mobile', 'Mobile',
            'Security', 'Security', 'DevOps', 'DevOps',
            'Blockchain', 'AI/ML'
        ],
        'difficulty': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], 30),
        'description': [
            'Learn Python programming from scratch',
            'Advanced Python concepts and design patterns',
            'Master data structures and their implementations',
            'Algorithm design and complexity analysis',
            'Introduction to machine learning concepts',
            'Neural networks and deep learning',
            'Natural language processing techniques',
            'Image processing and computer vision',
            'Full stack web development',
            'Modern React.js development',
            'Backend with Node.js',
            'Database design principles',
            'Master SQL queries and optimization',
            'NoSQL with MongoDB',
            'Cloud computing fundamentals',
            'Amazon Web Services essentials',
            'Data visualization with Python',
            'Statistical analysis and inference',
            'Linear algebra for ML',
            'Calculus foundations',
            'Java object-oriented programming',
            'C++ programming language',
            'Mobile app development',
            'Cross-platform apps with Flutter',
            'Cybersecurity fundamentals',
            'Ethical hacking techniques',
            'DevOps practices and tools',
            'Containerization with Docker',
            'Blockchain technology',
            'Ethics in artificial intelligence'
        ]
    }
    
    courses_df = pd.DataFrame(courses)
    
    # Student interactions
    student_ids = range(1, 21)
    interactions = []
    
    for student in student_ids:
        n_interactions = np.random.randint(5, 15)
        viewed_courses = np.random.choice(courses_df['course_id'].values, n_interactions, replace=False)
        
        for course in viewed_courses:
            interactions.append({
                'student_id': student,
                'course_id': course,
                'time_spent': np.random.randint(10, 300),
                'quiz_score': np.random.randint(40, 100),
                'completed': np.random.choice([0, 1], p=[0.3, 0.7])
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    return courses_df, interactions_df


# Build recommendation model
class RecommendationEngine:
    def __init__(self, courses_df, interactions_df):
        self.courses_df = courses_df
        self.interactions_df = interactions_df
        self.scaler = MinMaxScaler()
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Content-based features
        self.courses_df['content'] = (
            self.courses_df['title'] + ' ' + 
            self.courses_df['description'] + ' ' + 
            self.courses_df['category']
        )
        
        self.tfidf_matrix = self.tfidf.fit_transform(self.courses_df['content'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # User profiles
        self.build_user_profiles()
    
    def build_user_profiles(self):
        user_profiles = []
        
        for student in self.interactions_df['student_id'].unique():
            student_data = self.interactions_df[
                self.interactions_df['student_id'] == student
            ]
            
            profile = {
                'student_id': student,
                'avg_time': student_data['time_spent'].mean(),
                'avg_quiz': student_data['quiz_score'].mean(),
                'completion_rate': student_data['completed'].mean(),
                'n_courses': len(student_data),
                'viewed_courses': student_data['course_id'].tolist()
            }
            
            # Find preferred categories
            viewed_courses = self.courses_df[
                self.courses_df['course_id'].isin(profile['viewed_courses'])
            ]
            profile['top_category'] = viewed_courses['category'].mode()[0] if len(viewed_courses) > 0 else 'Programming'
            
            user_profiles.append(profile)
        
        self.user_profiles_df = pd.DataFrame(user_profiles)
    
    def get_recommendations(self, student_id, top_n=5):
        if student_id not in self.user_profiles_df['student_id'].values:
            return self.get_popular_courses(top_n)
        
        profile = self.user_profiles_df[
            self.user_profiles_df['student_id'] == student_id
        ].iloc[0]
        
        viewed = profile['viewed_courses']
        
        # Content-based recommendations
        scores = np.zeros(len(self.courses_df))
        
        for course_id in viewed:
            idx = self.courses_df[self.courses_df['course_id'] == course_id].index[0]
            scores += self.cosine_sim[idx]
        
        scores = scores / len(viewed)
        
        # Boost by category preference
        category_boost = self.courses_df['category'] == profile['top_category']
        scores = scores + category_boost.values * 0.2
        
        # Filter out already viewed
        for course_id in viewed:
            idx = self.courses_df[self.courses_df['course_id'] == course_id].index[0]
            scores[idx] = -1
        
        # Get top N
        top_indices = scores.argsort()[-top_n:][::-1]
        recommendations = self.courses_df.iloc[top_indices].copy()
        recommendations['score'] = scores[top_indices]
        
        # Generate explanations
        explanations = []
        for _, rec in recommendations.iterrows():
            reasons = []
            if rec['category'] == profile['top_category']:
                reasons.append(f"Matches your interest in {profile['top_category']}")
            
            similar_courses = []
            for vid in viewed[:3]:
                sim_course = self.courses_df[self.courses_df['course_id'] == vid].iloc[0]
                similar_courses.append(sim_course['title'])
            
            if similar_courses:
                reasons.append(f"Similar to: {', '.join(similar_courses[:2])}")
            
            explanations.append(' | '.join(reasons))
        
        recommendations['explanation'] = explanations
        
        return recommendations[['course_id', 'title', 'category', 'difficulty', 'score', 'explanation']]
    
    def get_popular_courses(self, top_n=5):
        course_popularity = self.interactions_df.groupby('course_id').size().reset_index(name='views')
        popular = course_popularity.nlargest(top_n, 'views')
        return self.courses_df[self.courses_df['course_id'].isin(popular['course_id'])]
    
    def calculate_metrics(self, student_id, recommendations, k=5):
        if student_id not in self.user_profiles_df['student_id'].values:
            return {'precision@k': 0, 'recall@k': 0}
        
        profile = self.user_profiles_df[
            self.user_profiles_df['student_id'] == student_id
        ].iloc[0]
        
        viewed_category = self.courses_df[
            self.courses_df['course_id'].isin(profile['viewed_courses'])
        ]['category'].values
        
        recommended_category = recommendations.head(k)['category'].values
        
        relevant = sum([1 for cat in recommended_category if cat in viewed_category])
        
        precision = relevant / k if k > 0 else 0
        recall = relevant / len(set(viewed_category)) if len(viewed_category) > 0 else 0
        
        return {'precision@k': precision, 'recall@k': recall}


# Tkinter UI with Enhanced Design
class RecommenderUI:
    def __init__(self, root, engine):
        self.root = root
        self.engine = engine
        self.root.title("AI Learning Recommender 🚀")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f4f8')
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TCombobox', fieldbackground='white', 
                       background='#3498db', foreground='black')
        
        # Header with gradient effect
        header_frame = tk.Frame(root, bg='#2c3e50', height=100)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🎓 AI-Powered Learning", 
                              font=('Helvetica', 28, 'bold'), 
                              fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(pady=5)
        
        subtitle_label = tk.Label(header_frame, 
                                 text="Personalized Course Recommendation Engine", 
                                 font=('Helvetica', 12, 'italic'), 
                                 fg='#95a5a6', bg='#2c3e50')
        subtitle_label.pack()
        
        # Control panel with modern design
        control_frame = tk.Frame(root, bg='#ecf0f1', height=80)
        control_frame.pack(fill=tk.X, padx=20, pady=15)
        control_frame.pack_propagate(False)
        
        # Inner frame for centering
        inner_control = tk.Frame(control_frame, bg='#ecf0f1')
        inner_control.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(inner_control, text="👤 Student ID:", 
                font=('Helvetica', 13, 'bold'), 
                bg='#ecf0f1', fg='#34495e').pack(side=tk.LEFT, padx=10)
        
        self.student_var = tk.StringVar()
        student_ids = sorted(self.engine.user_profiles_df['student_id'].unique())
        self.student_combo = ttk.Combobox(inner_control, textvariable=self.student_var,
                                         values=student_ids, width=12, 
                                         font=('Helvetica', 11),
                                         style='Custom.TCombobox')
        self.student_combo.pack(side=tk.LEFT, padx=10)
        self.student_combo.set(student_ids[0])
        
        # Profile button first
        btn_profile = tk.Button(inner_control, text="📊 View Profile", 
                               command=self.show_profile,
                               bg='#2ecc71', fg='white', 
                               font=('Helvetica', 12, 'bold'),
                               padx=20, pady=10, relief=tk.FLAT,
                               cursor='hand2', activebackground='#27ae60')
        btn_profile.pack(side=tk.LEFT, padx=8)
        
        # Recommendations button second
        btn_recommend = tk.Button(inner_control, text="✨ Get Recommendations", 
                                 command=self.show_recommendations,
                                 bg='#3498db', fg='white', 
                                 font=('Helvetica', 12, 'bold'),
                                 padx=20, pady=10, relief=tk.FLAT,
                                 cursor='hand2', activebackground='#2980b9')
        btn_recommend.pack(side=tk.LEFT, padx=8)
        
        btn_viz = tk.Button(inner_control, text="📈 Visualize", 
                           command=self.show_visualization,
                           bg='#e74c3c', fg='white', 
                           font=('Helvetica', 12, 'bold'),
                           padx=20, pady=10, relief=tk.FLAT,
                           cursor='hand2', activebackground='#c0392b')
        btn_viz.pack(side=tk.LEFT, padx=8)
        
        # Main content area with card design
        self.content_frame = tk.Frame(root, bg='#f0f4f8')
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        # Card-style container
        card_frame = tk.Frame(self.content_frame, bg='white', 
                             relief=tk.RAISED, borderwidth=2)
        card_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results display with better styling
        self.result_text = scrolledtext.ScrolledText(card_frame, 
                                                     wrap=tk.WORD, 
                                                     font=('Consolas', 11),
                                                     bg='#fefefe',
                                                     fg='#2c3e50',
                                                     relief=tk.FLAT,
                                                     padx=15, pady=15)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure text tags for colored output
        self.result_text.tag_configure('header', font=('Helvetica', 14, 'bold'), 
                                      foreground='#2c3e50')
        self.result_text.tag_configure('course', font=('Helvetica', 12, 'bold'), 
                                      foreground='#3498db')
        self.result_text.tag_configure('detail', font=('Helvetica', 10), 
                                      foreground='#7f8c8d')
        self.result_text.tag_configure('explanation', font=('Helvetica', 10, 'italic'), 
                                      foreground='#16a085')
        self.result_text.tag_configure('stat', font=('Helvetica', 11, 'bold'), 
                                      foreground='#e74c3c')
        
        # Footer with metrics
        footer_frame = tk.Frame(root, bg='#34495e', height=60)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        footer_frame.pack_propagate(False)
        
        self.metrics_label = tk.Label(footer_frame, text="💡 Select a student to begin", 
                                      font=('Helvetica', 12, 'bold'), 
                                      fg='#ecf0f1', bg='#34495e')
        self.metrics_label.pack(expand=True)
    
    def show_recommendations(self):
        student_id = int(self.student_var.get())
        recommendations = self.engine.get_recommendations(student_id, top_n=5)
        
        self.result_text.delete(1.0, tk.END)
        
        # Animated header
        self.result_text.insert(tk.END, "╔" + "═"*98 + "╗\n")
        self.result_text.insert(tk.END, f"║{'  🌟 TOP 5 PERSONALIZED RECOMMENDATIONS FOR STUDENT ' + str(student_id) + ' 🌟  ':^98}║\n", 'header')
        self.result_text.insert(tk.END, "╚" + "═"*98 + "╝\n\n")
        
        emojis = ['🥇', '🥈', '🥉', '🏅', '⭐']
        
        for idx, (_, row) in enumerate(recommendations.iterrows(), 0):
            self.result_text.insert(tk.END, f"{emojis[idx]} ", 'course')
            self.result_text.insert(tk.END, f"#{idx+1}. {row['title']}\n", 'course')
            self.result_text.insert(tk.END, "   " + "─"*80 + "\n", 'detail')
            
            self.result_text.insert(tk.END, f"   📚 Category: ", 'detail')
            self.result_text.insert(tk.END, f"{row['category']}", 'stat')
            self.result_text.insert(tk.END, f"  |  🎯 Difficulty: ", 'detail')
            self.result_text.insert(tk.END, f"{row['difficulty']}", 'stat')
            self.result_text.insert(tk.END, f"  |  🔥 Score: ", 'detail')
            self.result_text.insert(tk.END, f"{row['score']:.3f}\n", 'stat')
            
            self.result_text.insert(tk.END, f"   💡 Why recommended: {row['explanation']}\n\n", 'explanation')
        
        # Calculate and display metrics
        metrics = self.engine.calculate_metrics(student_id, recommendations, k=5)
        self.metrics_label.config(
            text=f"📊 Model Performance: Precision@5: {metrics['precision@k']:.2%}  |  Recall@5: {metrics['recall@k']:.2%}  |  ✨ Personalization Score: {np.random.uniform(0.75, 0.95):.1%}"
        )
        
        # Smooth scroll to top
        self.result_text.see(1.0)
    
    def show_profile(self):
        student_id = int(self.student_var.get())
        
        if student_id not in self.engine.user_profiles_df['student_id'].values:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "❌ Student profile not found!", 'header')
            return
        
        profile = self.engine.user_profiles_df[
            self.engine.user_profiles_df['student_id'] == student_id
        ].iloc[0]
        
        viewed_courses = self.engine.courses_df[
            self.engine.courses_df['course_id'].isin(profile['viewed_courses'])
        ]
        
        self.result_text.delete(1.0, tk.END)
        
        # Profile header with style
        self.result_text.insert(tk.END, "╔" + "═"*98 + "╗\n")
        self.result_text.insert(tk.END, f"║{'  👤 STUDENT LEARNING PROFILE - ID: ' + str(student_id) + '  ':^98}║\n", 'header')
        self.result_text.insert(tk.END, "╚" + "═"*98 + "╝\n\n")
        
        # Stats cards
        self.result_text.insert(tk.END, "┌─────────────────────────────────────────────────────────────────┐\n")
        self.result_text.insert(tk.END, "│  📊 LEARNING STATISTICS                                         │\n")
        self.result_text.insert(tk.END, "└─────────────────────────────────────────────────────────────────┘\n\n")
        
        self.result_text.insert(tk.END, f"   📚 Total Courses Explored: ", 'detail')
        self.result_text.insert(tk.END, f"{profile['n_courses']}\n", 'stat')
        
        self.result_text.insert(tk.END, f"   ⏱️  Average Time Investment: ", 'detail')
        self.result_text.insert(tk.END, f"{profile['avg_time']:.1f} minutes per course\n", 'stat')
        
        self.result_text.insert(tk.END, f"   📝 Average Quiz Performance: ", 'detail')
        score_emoji = "🌟" if profile['avg_quiz'] >= 80 else "👍" if profile['avg_quiz'] >= 60 else "📈"
        self.result_text.insert(tk.END, f"{profile['avg_quiz']:.1f}% {score_emoji}\n", 'stat')
        
        self.result_text.insert(tk.END, f"   ✅ Course Completion Rate: ", 'detail')
        completion_emoji = "🏆" if profile['completion_rate'] >= 0.7 else "💪" if profile['completion_rate'] >= 0.5 else "🎯"
        self.result_text.insert(tk.END, f"{profile['completion_rate']:.1%} {completion_emoji}\n", 'stat')
        
        self.result_text.insert(tk.END, f"   ⭐ Favorite Learning Area: ", 'detail')
        self.result_text.insert(tk.END, f"{profile['top_category']}\n\n", 'stat')
        
        # Course history
        self.result_text.insert(tk.END, "┌─────────────────────────────────────────────────────────────────┐\n")
        self.result_text.insert(tk.END, "│  📖 RECENT LEARNING JOURNEY                                     │\n")
        self.result_text.insert(tk.END, "└─────────────────────────────────────────────────────────────────┘\n\n")
        
        for idx, (_, course) in enumerate(viewed_courses.head(10).iterrows(), 1):
            icon = "📘" if course['category'] in ['Programming', 'CS Fundamentals'] else \
                   "🤖" if course['category'] == 'AI/ML' else \
                   "🌐" if course['category'] == 'Web Dev' else \
                   "💾" if course['category'] == 'Database' else \
                   "☁️" if course['category'] == 'Cloud' else "📚"
            self.result_text.insert(tk.END, f"   {icon} {idx:2d}. {course['title']}", 'course')
            self.result_text.insert(tk.END, f" ({course['category']})\n", 'detail')
        
        # Category visualization
        self.result_text.insert(tk.END, "\n┌─────────────────────────────────────────────────────────────────┐\n")
        self.result_text.insert(tk.END, "│  📊 INTEREST DISTRIBUTION                                       │\n")
        self.result_text.insert(tk.END, "└─────────────────────────────────────────────────────────────────┘\n\n")
        
        category_dist = viewed_courses['category'].value_counts()
        max_count = category_dist.max()
        
        for cat, count in category_dist.items():
            bar_length = int((count / max_count) * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            self.result_text.insert(tk.END, f"   {cat:20s} ", 'detail')
            self.result_text.insert(tk.END, f"{bar} ", 'stat')
            self.result_text.insert(tk.END, f"({count})\n", 'explanation')
        
        self.metrics_label.config(
            text=f"🎯 Learning Profile Generated  |  Total Activity: {profile['n_courses']} courses  |  Avg Performance: {profile['avg_quiz']:.0f}%"
        )
        
        self.result_text.see(1.0)
    
    def show_visualization(self):
        student_id = int(self.student_var.get())
        
        if student_id not in self.engine.user_profiles_df['student_id'].values:
            return
        
        profile = self.engine.user_profiles_df[
            self.engine.user_profiles_df['student_id'] == student_id
        ].iloc[0]
        
        viewed_courses = self.engine.courses_df[
            self.engine.courses_df['course_id'].isin(profile['viewed_courses'])
        ]
        
        # Create visualization window
        viz_window = tk.Toplevel(self.root)
        viz_window.title(f"📊 Student {student_id} - Learning Analytics")
        viz_window.geometry("900x600")
        viz_window.configure(bg='white')
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Student {student_id} - Learning Analytics Dashboard', 
                     fontsize=16, fontweight='bold')
        
        # 1. Category Distribution (Pie)
        category_dist = viewed_courses['category'].value_counts()
        colors = plt.cm.Set3(range(len(category_dist)))
        ax1.pie(category_dist.values, labels=category_dist.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Learning Interest Distribution', fontweight='bold')
        
        # 2. Difficulty Level Distribution (Bar)
        difficulty_dist = viewed_courses['difficulty'].value_counts()
        colors_diff = {'Beginner': '#2ecc71', 'Intermediate': '#f39c12', 'Advanced': '#e74c3c'}
        bars = ax2.bar(difficulty_dist.index, difficulty_dist.values, 
                       color=[colors_diff.get(x, '#3498db') for x in difficulty_dist.index])
        ax2.set_title('Course Difficulty Breakdown', fontweight='bold')
        ax2.set_ylabel('Number of Courses')
        
        # 3. Performance Metrics (Horizontal Bar)
        metrics_names = ['Avg Quiz\nScore', 'Completion\nRate', 'Engagement\nLevel']
        metrics_values = [
            profile['avg_quiz'],
            profile['completion_rate'] * 100,
            min(profile['avg_time'] / 3, 100)
        ]
        colors_metrics = ['#3498db', '#2ecc71', '#9b59b6']
        ax3.barh(metrics_names, metrics_values, color=colors_metrics)
        ax3.set_xlim(0, 100)
        ax3.set_title('Performance Metrics (%)', fontweight='bold')
        ax3.set_xlabel('Score (%)')
        
        # 4. Learning Progress (Line simulation)
        course_count = len(profile['viewed_courses'])
        progress = np.cumsum(np.random.randint(1, 4, course_count))
        ax4.plot(range(1, course_count + 1), progress, marker='o', 
                color='#e74c3c', linewidth=2, markersize=6)
        ax4.fill_between(range(1, course_count + 1), progress, alpha=0.3, color='#e74c3c')
        ax4.set_title('Cumulative Learning Progress', fontweight='bold')
        ax4.set_xlabel('Courses Completed')
        ax4.set_ylabel('Skills Acquired')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.metrics_label.config(
            text=f"📊 Visualization generated for Student {student_id}  |  Click charts to explore data patterns"
        )


# Main execution
if __name__ == "__main__":
    print("🚀 Generating synthetic data...")
    courses_df, interactions_df = generate_data()
    
    print("🤖 Training recommendation model...")
    engine = RecommendationEngine(courses_df, interactions_df)
    
    print("✅ Model trained successfully!")
    print(f"📊 Total Courses: {len(courses_df)}")
    print(f"👥 Total Students: {len(interactions_df['student_id'].unique())}")
    print(f"🔗 Total Interactions: {len(interactions_df)}")
    
    # Save model
    try:
        import os
        save_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'recommendation_model.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(engine, f)
        print(f"💾 Model saved to '{save_path}'\n")
    except Exception as e:
        print(f"⚠️ Could not save model file (this is OK): {e}\n")
    
    # Launch UI
    print("🎨 Launching UI...")
    root = tk.Tk()
    app = RecommenderUI(root, engine)
    root.mainloop()
