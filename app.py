import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime, timedelta
import requests
from typing import Dict, List

# ---------- Configuration ----------
class Config:
    DATA_PATH = "data/transactions.csv"
    GOALS_PATH = "data/goals.json"
    OLLAMA_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.2"
    
    os.makedirs("data", exist_ok=True)

class AIAssistant:
    def __init__(self):
        self.base_url = Config.OLLAMA_URL
        self.model = Config.DEFAULT_MODEL

    def get_advice(self, income: float, expenses: float, savings: float) -> str:
        """Provide simple financial advice based on income, expenses, and savings."""
        if income <= 0:
            return "Add your income to get personalized advice."
        savings_rate = (savings / income) * 100 if income > 0 else 0
        expense_ratio = (expenses / income) * 100 if income > 0 else 0
        advice = ""
        if savings_rate >= 30:
            advice = "Excellent savings rate! Consider investing for long-term growth."
        elif savings_rate >= 20:
            advice = "Good savings rate! Build an emergency fund and start investing."
        elif savings_rate >= 10:
            advice = "You're saving, but try to increase your savings rate to 20%."
        else:
            advice = "Focus on reducing expenses and increasing your savings rate."
        if expense_ratio > 90:
            advice += " Warning: Your expenses are very high compared to your income."
        elif expense_ratio > 80:
            advice += " Try to cut down on unnecessary expenses."
        return advice
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def get_dashboard_insights(self, df: pd.DataFrame, metrics: Dict) -> Dict:
        """Generate comprehensive insights from dashboard data"""
        insights = {}
        
        if df.empty:
            return {"general": "Start tracking your expenses to get AI insights!", "alerts": [], "tips": []}
        
        # Analyze spending patterns
        spending_analysis = self._analyze_spending_patterns(df)
        trend_analysis = self._analyze_trends(df)
        category_insights = self._analyze_categories(df)
        
        # Generate AI advice
        if self.is_available():
            insights = self._get_ai_insights(df, metrics, spending_analysis, trend_analysis)
        else:
            insights = self._get_smart_insights(df, metrics, spending_analysis, trend_analysis)
        
        return {
            "general": insights.get("advice", ""),
            "alerts": insights.get("alerts", []),
            "tips": insights.get("tips", []),
            "spending_pattern": spending_analysis,
            "trends": trend_analysis,
            "categories": category_insights
        }
    
    def _analyze_spending_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze spending patterns from transaction data"""
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.day_name()
        df['Date'] = df['DateTime'].dt.date
        
        expense_df = df[df['Type'] == 'Expense']
        
        if expense_df.empty:
            return {"peak_spending_time": None, "peak_spending_day": None, "daily_avg": 0}
        
        # Peak spending time and day
        hourly_spending = expense_df.groupby('Hour')['Amount'].sum()
        daily_spending = expense_df.groupby('DayOfWeek')['Amount'].sum()
        
        # Daily average
        daily_totals = expense_df.groupby('Date')['Amount'].sum()
        daily_avg = daily_totals.mean() if not daily_totals.empty else 0
        
        return {
            "peak_spending_time": hourly_spending.idxmax() if not hourly_spending.empty else None,
            "peak_spending_day": daily_spending.idxmax() if not daily_spending.empty else None,
            "daily_avg": daily_avg,
            "highest_expense": expense_df.loc[expense_df['Amount'].idxmax()] if not expense_df.empty else None
        }
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze financial trends"""
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Month'] = df['DateTime'].dt.strftime('%Y-%m')
        
        monthly_data = df.groupby(['Month', 'Type'])['Amount'].sum().unstack(fill_value=0)
        
        trends = {}
        for col in ['Income', 'Expense', 'Saving']:
            if col in monthly_data.columns and len(monthly_data) >= 2:
                current = monthly_data[col].iloc[-1]
                previous = monthly_data[col].iloc[-2]
                change = ((current - previous) / previous * 100) if previous > 0 else 0
                trends[col.lower() + '_trend'] = change
            else:
                trends[col.lower() + '_trend'] = 0
        
        return trends
    
    def _analyze_categories(self, df: pd.DataFrame) -> Dict:
        """Analyze category-wise spending"""
        expense_df = df[df['Type'] == 'Expense']
        if expense_df.empty:
            return {"top_categories": {}, "category_percentage": {}}
        
        category_spending = expense_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        total_expenses = category_spending.sum()
        
        category_percentage = (category_spending / total_expenses * 100).round(1)
        
        return {
            "top_categories": category_spending.head(3).to_dict(),
            "category_percentage": category_percentage.to_dict(),
            "highest_category": category_spending.index[0] if not category_spending.empty else None
        }
    
    def _get_ai_insights(self, df: pd.DataFrame, metrics: Dict, spending_analysis: Dict, trend_analysis: Dict) -> Dict:
        """Get AI-powered insights"""
        try:
            # Create comprehensive prompt
            prompt = f"""
            Analyze this financial dashboard data and provide insights:
            
            FINANCIAL SUMMARY:
            - Income: ₹{metrics['income']:,.0f}
            - Expenses: ₹{metrics['expenses']:,.0f}
            - Savings: ₹{metrics['savings']:,.0f}
            - Balance: ₹{metrics['balance']:,.0f}
            
            SPENDING PATTERNS:
            - Peak spending time: {spending_analysis.get('peak_spending_time', 'N/A')}:00
            - Peak spending day: {spending_analysis.get('peak_spending_day', 'N/A')}
            - Daily average expense: ₹{spending_analysis.get('daily_avg', 0):,.0f}
            
            TRENDS:
            - Expense trend: {trend_analysis.get('expense_trend', 0):+.1f}%
            - Income trend: {trend_analysis.get('income_trend', 0):+.1f}%
            - Saving trend: {trend_analysis.get('saving_trend', 0):+.1f}%
            
            Provide:
            1. One key insight about spending behavior
            2. One alert if there's a concern
            3. Two actionable tips for improvement
            
            Format: Keep each point under 50 words, practical and encouraging.
            """
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(f"{self.base_url}/api/generate", 
                                   json=payload, timeout=20)
            
            if response.status_code == 200:
                ai_response = response.json()['response'].strip()
                return self._parse_ai_response(ai_response)
        except:
            pass
        
        return self._get_smart_insights(df, metrics, spending_analysis, trend_analysis)
    
    def _parse_ai_response(self, response: str) -> Dict:
        """Parse AI response into structured format"""
        lines = response.split('\n')
        return {
            "advice": response[:200] + "..." if len(response) > 200 else response,
            "alerts": [],
            "tips": []
        }
    
    def _get_smart_insights(self, df: pd.DataFrame, metrics: Dict, spending_analysis: Dict, trend_analysis: Dict) -> Dict:
        """Generate smart insights when AI is not available"""
        alerts = []
        tips = []
        advice = ""
        
        # Calculate key ratios
        savings_rate = (metrics['savings'] / metrics['income'] * 100) if metrics['income'] > 0 else 0
        expense_ratio = (metrics['expenses'] / metrics['income'] * 100) if metrics['income'] > 0 else 0
        
        # Generate advice based on spending patterns
        peak_time = spending_analysis.get('peak_spending_time')
        peak_day = spending_analysis.get('peak_spending_day')
        daily_avg = spending_analysis.get('daily_avg', 0)
        
        if peak_time and peak_day:
            advice = f"📊 You spend most on {peak_day}s around {peak_time}:00. Daily average: ₹{daily_avg:,.0f}. "
        
        # Savings rate analysis
        if savings_rate >= 30:
            advice += "🌟 Excellent savings rate! Consider investing for wealth growth."
        elif savings_rate >= 20:
            advice += "👍 Good savings rate! Build emergency fund worth 6 months expenses."
        elif savings_rate >= 10:
            advice += "📈 You're saving! Try the 50-30-20 rule for better financial health."
        else:
            advice += "💡 Focus on increasing your savings rate to at least 10%."
            alerts.append("⚠️ Low savings rate - need to cut expenses or increase income")
        
        # Expense ratio alerts
        if expense_ratio > 90:
            alerts.append("🚨 Spending over 90% of income - immediate action needed!")
        elif expense_ratio > 80:
            alerts.append("⚠️ High expense ratio - review unnecessary spending")
        
        # Trend-based tips
        expense_trend = trend_analysis.get('expense_trend', 0)
        if expense_trend > 20:
            alerts.append(f"📈 Expenses increased by {expense_trend:.1f}% - monitor closely")
            tips.append("📊 Track daily expenses to identify spending triggers")
        
        # General tips based on data
        if metrics['expenses'] > 0:
            tips.append(f"💰 Try saving ₹{daily_avg*0.1:.0f} daily to improve financial health")
        
        if len(df[df['Type'] == 'Expense']) > 10:
            tips.append("📱 Use expense categories to identify your biggest spending areas")
        else:
            tips.append("📝 Track more transactions to get better insights")
        
        return {
            "advice": advice,
            "alerts": alerts,
            "tips": tips
        }

# ---------- Data Management ----------
def load_data() -> pd.DataFrame:
    if os.path.exists(Config.DATA_PATH):
        return pd.read_csv(Config.DATA_PATH)
    return pd.DataFrame(columns=["DateTime", "Type", "Category", "Amount", "Description"])

def save_transaction(transaction_data: Dict):
    df = load_data()
    new_row = pd.DataFrame([transaction_data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(Config.DATA_PATH, index=False)

def calculate_metrics(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {'income': 0, 'expenses': 0, 'savings': 0, 'balance': 0}
    
    income = df[df['Type'] == 'Income']['Amount'].sum()
    expenses = df[df['Type'] == 'Expense']['Amount'].sum()
    savings = df[df['Type'] == 'Saving']['Amount'].sum()
    balance = income - expenses
    
    return {
        'income': income,
        'expenses': expenses, 
        'savings': savings,
        'balance': balance
    }

# ---------- Visualization ----------
def create_dashboard_chart(df: pd.DataFrame):
    if df.empty:
        return None
    
    # Monthly trends
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Month'] = df['DateTime'].dt.strftime('%Y-%m')
    
    monthly_data = df.groupby(['Month', 'Type'])['Amount'].sum().unstack(fill_value=0)
    
    fig = go.Figure()
    
    colors = {'Income': '#2E8B57', 'Expense': '#DC143C', 'Saving': '#4169E1'}
    
    for col in monthly_data.columns:
        if col in colors:
            fig.add_trace(go.Scatter(
                x=monthly_data.index,
                y=monthly_data[col],
                name=col,
                line=dict(color=colors[col], width=3),
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title="📈 Monthly Financial Trends",
        xaxis_title="Month",
        yaxis_title="Amount (₹)",
        height=400,
        showlegend=True
    )
    
    return fig

def create_category_chart(df: pd.DataFrame):
    if df.empty or df[df['Type'] == 'Expense'].empty:
        return None
    
    expense_data = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum().sort_values(ascending=False)
    
    fig = px.pie(
        values=expense_data.values,
        names=expense_data.index,
        title="💸 Expense Breakdown by Category"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

# ---------- Main App ----------
def main():
    st.set_page_config(
        page_title="Wealthic",
        page_icon="💰",
        layout="wide"
    )
    
    # Custom CSS for modern look
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .ai-response {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Wealthic</h1>
        <p>AI-Powered Personal Finance Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize AI
    ai_assistant = AIAssistant()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Quick Actions")
        
        if ai_assistant.is_available():
            st.success("🤖 AI Assistant: Online")
        else:
            st.info("🤖 AI Assistant: Offline Mode")
            st.caption("Install Ollama for AI features")
        
        # Quick stats
        df = load_data()
        if not df.empty:
            metrics = calculate_metrics(df)
            st.metric("💳 Current Balance", f"₹{metrics['balance']:,.0f}")
            st.metric("📊 Total Transactions", len(df))
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "➕ Add Transaction", "🤖 AI Insights"])
    
    with tab1:
        df = load_data()
        
        if df.empty:
            st.info("🚀 Start by adding your first transaction!")
            st.markdown("""
            ### 🎯 Get Started:
            1. Click **"Add Transaction"** tab
            2. Enter your income/expenses
            3. Watch AI analyze your spending patterns!
            """)
        else:
            # Calculate metrics
            metrics = calculate_metrics(df)
            
            # Get AI insights from dashboard data
            with st.spinner("🧠 AI is analyzing your financial patterns..."):
                insights = ai_assistant.get_dashboard_insights(df, metrics)
            
            # Main metrics with AI insights overlay
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Income", f"₹{metrics['income']:,.0f}")
            with col2:
                st.metric("💸 Expenses", f"₹{metrics['expenses']:,.0f}")
            with col3:
                st.metric("🏦 Savings", f"₹{metrics['savings']:,.0f}")
            with col4:
                st.metric("📈 Balance", f"₹{metrics['balance']:,.0f}")
            
            # AI Insights Panel - Prominently displayed
            st.markdown("### 🤖 Live AI Analysis")
            
            # Main AI advice
            if insights.get('advice'):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
                    <h4>🎯 Smart Insights</h4>
                    <p style="margin: 0; font-size: 1.1em;">{insights['advice']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Alerts and Tips in columns
            col1, col2 = st.columns(2)
            
            with col1:
                if insights.get('alerts'):
                    st.markdown("#### ⚠️ Alerts")
                    for alert in insights['alerts'][:3]:  # Show top 3 alerts
                        st.warning(alert)
            
            with col2:
                if insights.get('tips'):
                    st.markdown("#### 💡 Smart Tips")
                    for tip in insights['tips'][:3]:  # Show top 3 tips
                        st.info(tip)
            
            # Spending Pattern Insights
            if insights.get('spending_pattern'):
                pattern = insights['spending_pattern']
                if pattern.get('peak_spending_time') and pattern.get('peak_spending_day'):
                    st.markdown("### 🕐 Your Spending Pattern")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🕐 Peak Time", f"{pattern['peak_spending_time']}:00")
                    with col2:
                        st.metric("📅 Peak Day", pattern['peak_spending_day'])
                    with col3:
                        st.metric("📊 Daily Avg", f"₹{pattern['daily_avg']:,.0f}")
            
            # Charts with AI annotations
            col1, col2 = st.columns(2)
            
            with col1:
                trend_chart = create_dashboard_chart(df)
                if trend_chart:
                    st.plotly_chart(trend_chart, use_container_width=True)
                    
                    # Add AI trend insights
                    if insights.get('trends'):
                        trends = insights['trends']
                        expense_trend = trends.get('expense_trend', 0)
                        if abs(expense_trend) > 10:
                            trend_color = "🔴" if expense_trend > 0 else "🟢"
                            st.caption(f"{trend_color} Expense trend: {expense_trend:+.1f}% vs last month")
            
            with col2:
                category_chart = create_category_chart(df)
                if category_chart:
                    st.plotly_chart(category_chart, use_container_width=True)
                    
                    # Add AI category insights
                    if insights.get('categories', {}).get('highest_category'):
                        highest_cat = insights['categories']['highest_category']
                        percentage = insights['categories']['category_percentage'].get(highest_cat, 0)
                        st.caption(f"🎯 Highest spending: {highest_cat} ({percentage:.1f}%)")
            
            # Interactive AI Chat in Dashboard
            st.markdown("### 💬 Ask AI About Your Dashboard")
            
            if "dashboard_chat" not in st.session_state:
                st.session_state.dashboard_chat = []
            
            # Quick question buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💰 How can I save more?"):
                    question = f"Based on my spending of ₹{metrics['expenses']:,.0f} and savings of ₹{metrics['savings']:,.0f}, how can I save more money?"
                    with st.spinner("AI thinking..."):
                        answer = ai_assistant.get_advice(metrics['income'], metrics['expenses'], metrics['savings'])
                    st.session_state.dashboard_chat.append({"q": "How can I save more?", "a": answer})
            
            with col2:
                if st.button("📊 Analyze my trends"):
                    if insights.get('trends'):
                        trends = insights['trends']
                        answer = f"Your expense trend is {trends.get('expense_trend', 0):+.1f}%, income trend is {trends.get('income_trend', 0):+.1f}%. "
                        if trends.get('expense_trend', 0) > 10:
                            answer += "Your expenses are rising - consider budgeting and tracking daily spends."
                        else:
                            answer += "Your spending is stable - good financial discipline!"
                    else:
                        answer = "Add more transactions to see trend analysis."
                    st.session_state.dashboard_chat.append({"q": "Analyze my trends", "a": answer})
            
            with col3:
                if st.button("🎯 Budget suggestions"):
                    savings_rate = (metrics['savings'] / metrics['income'] * 100) if metrics['income'] > 0 else 0
                    answer = f"With {savings_rate:.1f}% savings rate, try 50-30-20 budgeting: ₹{metrics['income']*0.5:,.0f} needs, ₹{metrics['income']*0.3:,.0f} wants, ₹{metrics['income']*0.2:,.0f} savings."
                    st.session_state.dashboard_chat.append({"q": "Budget suggestions", "a": answer})
            
            # Show recent chat
            if st.session_state.dashboard_chat:
                st.markdown("#### 💬 Recent AI Responses")
                for chat in st.session_state.dashboard_chat[-2:]:  # Show last 2
                    with st.expander(f"❓ {chat['q']}", expanded=False):
                        st.info(chat['a'])
            
            # Recent transactions with AI highlights
            st.markdown("### 📋 Recent Transactions")
            recent_df = df.tail(10).iloc[::-1]  # Last 10, reversed
            
            # Highlight unusual transactions
            if not recent_df.empty and insights.get('spending_pattern', {}).get('daily_avg'):
                daily_avg = insights['spending_pattern']['daily_avg']
                unusual_transactions = recent_df[
                    (recent_df['Type'] == 'Expense') & 
                    (recent_df['Amount'] > daily_avg * 2)
                ]
                
                if not unusual_transactions.empty:
                    st.warning(f"🔍 AI noticed {len(unusual_transactions)} high-value transactions above ₹{daily_avg*2:,.0f}")
            
            st.dataframe(
                recent_df[['DateTime', 'Type', 'Category', 'Amount', 'Description']], 
                use_container_width=True,
                hide_index=True
            )
    
    with tab2:
        st.markdown("### ➕ Add New Transaction")
        
        with st.form("transaction_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                trans_type = st.selectbox("Type", ["Income", "Expense", "Saving"])
            
            with col2:
                # Smart categories based on type
                categories = {
                    'Income': ['Salary', 'Freelance', 'Investment Returns', 'Gift', 'Other'],
                    'Expense': ['Food & Dining', 'Transportation', 'Shopping', 'Bills & Utilities', 'Entertainment', 'Healthcare', 'Other'],
                    'Saving': ['Emergency Fund', 'Investment', 'Fixed Deposit', 'Goal Saving', 'Other']
                }
                category = st.selectbox("Category", categories[trans_type])
            
            amount = st.number_input("Amount (₹)", min_value=1.0, step=1.0)
            description = st.text_input("Description", placeholder="What was this for?")
            
            submitted = st.form_submit_button("💾 Add Transaction", use_container_width=True)
            
            if submitted and amount > 0:
                transaction = {
                    'DateTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Type': trans_type,
                    'Category': category,
                    'Amount': amount,
                    'Description': description
                }
                
                save_transaction(transaction)
                st.success("✅ Transaction added successfully!")
                st.rerun()
    
    with tab3:
        st.markdown("### 🤖  MAINAK AI Financial Advisor Chat")
        
        df = load_data()
        
        if df.empty:
            st.info("Add some transactions first to get personalized AI advice!")
        else:
            metrics = calculate_metrics(df)
            
            # Initialize chat history
            if "ai_chat_history" not in st.session_state:
                st.session_state.ai_chat_history = []
            
            # Chat interface
            st.markdown("#### 💬 Chat with Your Financial Advisor")
            
            # Display chat history
            for message in st.session_state.ai_chat_history[-8:]:  # Show last 8 messages
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
            
            # Chat input
            user_question = st.chat_input("Ask anything about your finances...")
            
            if user_question:
                # Add user message
                st.session_state.ai_chat_history.append({"role": "user", "content": user_question})
                
                # Show user message
                with st.chat_message("user"):
                    st.write(user_question)
                
                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("AI is thinking..."):
                        # Create context from current data
                        context = f"""
                        Current Financial Status:
                        - Income: ₹{metrics['income']:,.0f}
                        - Expenses: ₹{metrics['expenses']:,.0f}
                        - Savings: ₹{metrics['savings']:,.0f}
                        - Balance: ₹{metrics['balance']:,.0f}
                        
                        Question: {user_question}
                        """
                        
                        if ai_assistant.is_available():
                            try:
                                payload = {
                                    "model": ai_assistant.model,
                                    "prompt": f"As a financial advisor, answer this question based on the user's financial data: {context}. Give practical, actionable advice in 2-3 sentences.",
                                    "stream": False
                                }
                                
                                response = requests.post(f"{ai_assistant.base_url}/api/generate", 
                                                       json=payload, timeout=20)
                                
                                if response.status_code == 200:
                                    ai_response = response.json()['response'].strip()
                                else:
                                    ai_response = ai_assistant._get_smart_insights(df, metrics, {}, {})['advice']
                            except:
                                ai_response = "I'm having trouble connecting. Here's what I can tell you based on your data: " + ai_assistant._get_smart_insights(df, metrics, {}, {})['advice']
                        else:
                            # Smart fallback response
                            savings_rate = (metrics['savings'] / metrics['income'] * 100) if metrics['income'] > 0 else 0
                            
                            if "save" in user_question.lower():
                                ai_response = f"With your current {savings_rate:.1f}% savings rate, try automating ₹{metrics['income']*0.05:,.0f} monthly transfers to savings. Small consistent amounts build wealth over time!"
                            elif "budget" in user_question.lower():
                                ai_response = f"Based on your ₹{metrics['expenses']:,.0f} expenses, try the 50-30-20 rule: ₹{metrics['income']*0.5:,.0f} for needs, ₹{metrics['income']*0.3:,.0f} for wants, ₹{metrics['income']*0.2:,.0f} for savings."
                            elif "invest" in user_question.lower():
                                if savings_rate > 20:
                                    ai_response = "Great savings rate! Consider SIP mutual funds starting with ₹1000/month. Diversify between equity and debt funds based on your risk appetite."
                                else:
                                    ai_response = "Focus on building an emergency fund first (6 months expenses). Once that's done, start investing in mutual funds through SIP."
                            else:
                                ai_response = f"Looking at your finances: You're spending ₹{metrics['expenses']:,.0f} and saving ₹{metrics['savings']:,.0f}. Focus on tracking expenses daily and increasing savings gradually by 2-3% each month."
                        
                        st.write(ai_response)
                
                # Add AI response to history
                st.session_state.ai_chat_history.append({"role": "assistant", "content": ai_response})
            
            # Quick action buttons
            st.markdown("#### 🎯 Quick Financial Questions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💰 How to save more?", key="save_more"):
                    question = "How can I save more money from my current income?"
                    st.session_state.ai_chat_history.append({"role": "user", "content": question})
                    
                    savings_rate = (metrics['savings'] / metrics['income'] * 100) if metrics['income'] > 0 else 0
                    if savings_rate < 10:
                        answer = f"Start with baby steps: Save ₹{metrics['income']*0.02:,.0f} monthly (2% of income). Cut one subscription or eating out session. Track every expense for a week to find money leaks!"
                    else:
                        answer = f"You're already saving {savings_rate:.1f}%! Try increasing to {savings_rate+5:.1f}% by setting up automatic transfers. Challenge yourself to live on ₹{(metrics['expenses']-metrics['expenses']*0.1):,.0f} for a month."
                    
                    st.session_state.ai_chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
            
            with col2:
                if st.button("📊 Investment advice?", key="invest_advice"):
                    question = "Should I start investing? What's the best strategy?"
                    st.session_state.ai_chat_history.append({"role": "user", "content": question})
                    
                    if metrics['savings'] > metrics['expenses'] * 3:  # 3 months emergency fund
                        answer = "Perfect timing! You have emergency funds. Start SIP in diversified mutual funds with ₹2000-5000/month. 70% equity funds for growth, 30% debt funds for stability."
                    else:
                        answer = f"First, build emergency fund of ₹{metrics['expenses']*6:,.0f} (6 months expenses). Keep it in savings account or FD. Then start investing in mutual funds through SIP."
                    
                    st.session_state.ai_chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
            
            with col3:
                if st.button("🎯 Budget planning?", key="budget_plan"):
                    question = "Help me create a realistic budget plan"
                    st.session_state.ai_chat_history.append({"role": "user", "content": question})
                    
                    answer = f"""Perfect! Here's your personalized 50-30-20 budget:
                    
🏠 **Needs (50%)**: ₹{metrics['income']*0.5:,.0f} - Rent, groceries, utilities, EMIs
💫 **Wants (30%)**: ₹{metrics['income']*0.3:,.0f} - Entertainment, shopping, dining out  
💰 **Savings (20%)**: ₹{metrics['income']*0.2:,.0f} - Emergency fund + investments

Current spending: ₹{metrics['expenses']:,.0f}. You're {'under' if metrics['expenses'] < metrics['income']*0.8 else 'over'} the 80% limit. Track daily for 2 weeks to optimize!"""
                    
                    st.session_state.ai_chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
            
            # Clear chat button
            if st.session_state.ai_chat_history:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.ai_chat_history = []
                    st.rerun()(df)
            with st.spinner("AI is analyzing your finances..."):
                advice = ai_assistant.get_advice(
                    metrics['income'], 
                    metrics['expenses'], 
                    metrics['savings']
                )
            
            st.markdown(f"""
            <div class="ai-response">
                <h4>🎯 Personalized Financial Advice</h4>
                <p>{advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick insights
            st.markdown("#### 📊 Quick Analysis")
            
            if metrics['income'] > 0:
                savings_rate = (metrics['savings'] / metrics['income']) * 100
                expense_ratio = (metrics['expenses'] / metrics['income']) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("💹 Savings Rate", f"{savings_rate:.1f}%", 
                             delta="Excellent" if savings_rate >= 20 else "Needs Work")
                
                with col2:
                    st.metric("💳 Expense Ratio", f"{expense_ratio:.1f}%",
                             delta="Good" if expense_ratio <= 70 else "High")
                
                with col3:
                    if not df.empty:
                        avg_expense = metrics['expenses'] / len(df[df['Type'] == 'Expense']) if len(df[df['Type'] == 'Expense']) > 0 else 0
                        st.metric("📱 Avg Transaction", f"₹{avg_expense:.0f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        💰 Smart Wealth Tracker | AI-Powered Financial Management | Built BY MAINAK BOSE 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
