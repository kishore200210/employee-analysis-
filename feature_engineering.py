import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Engagement Score
        engagement_features = [
            'JobSatisfaction',
            'EnvironmentSatisfaction',
            'RelationshipSatisfaction',
            'WorkLifeBalance',
            'JobInvolvement'
        ]
        X['EngagementScore'] = X[engagement_features].mean(axis=1)

        # Promotion Wait
        X['PromotionWait'] = X['YearsSinceLastPromotion'] / (X['YearsAtCompany'] + 1)

        # Manager Stability
        X['ManagerStability'] = X['YearsWithCurrManager'] / (X['YearsAtCompany'] + 1)

        # Salary Stress
        X['SalaryStress'] = X['MonthlyIncome'] / (X['PercentSalaryHike'] + 1)

        # Travel Intensity
        X['TravelIntensity'] = (
            X.get('BusinessTravel_Travel_Frequently', 0) * 2 +
            X.get('BusinessTravel_Travel_Rarely', 0)
        )

        return X
