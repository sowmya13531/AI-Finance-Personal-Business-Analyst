def calculate_profit(revenue_df, expense_df):

    total_revenue = revenue_df["revenue"].sum()

    total_expenses = expense_df["amount"].sum()

    profit = total_revenue - total_expenses

    return profit