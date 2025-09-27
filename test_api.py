#!/usr/bin/env python3
"""
Script de test pour l'API DataTalk
Tests des endpoints principaux
"""

import requests
import json
import pandas as pd
import io
from pathlib import Path
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"  # Remplacer par votre clé API

class DataTalkAPITester:
    """Testeur pour l'API DataTalk"""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.session_id = "test-session-" + str(int(time.time()))
    
    def test_health(self):
        """Test du endpoint de santé"""
        print("🔍 Test de santé de l'API...")
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ API en bonne santé")
                return True
            else:
                print(f"❌ Problème de santé: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Erreur de connexion: {e}")
            return False
    
    def create_test_data(self):
        """Crée un dataset de test simple"""
        data = {
            'nom': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salaire': [50000, 60000, 70000, 55000, 65000],
            'departement': ['IT', 'Marketing', 'IT', 'RH', 'Marketing'],
            'experience': [2, 5, 10, 3, 7]
        }
        df = pd.DataFrame(data)
        
        # Convertir en CSV pour l'upload
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
    
    def test_data_upload(self):
        """Test d'upload de données"""
        print("\n📤 Test d'upload de données...")
        try:
            csv_data = self.create_test_data()
            
            files = {
                'file': ('test_data.csv', csv_data, 'text/csv')
            }
            data = {
                'session_id': self.session_id
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/data/upload",
                files=files,
                data=data,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Upload réussi: {result['rows']} lignes, {result['columns']} colonnes")
                print(f"📊 Colonnes: {', '.join(result['column_names'])}")
                return True
            else:
                print(f"❌ Échec upload: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur upload: {e}")
            return False
    
    def test_chat_query(self):
        """Test de requête en langage naturel"""
        print("\n💬 Test de requête NLQ...")
        try:
            payload = {
                "session_id": self.session_id,
                "query": "Quel est le salaire moyen par département?",
                "context": "Analyse des salaires"
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/chat/query",
                json=payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Requête traitée avec succès")
                print(f"🤖 Réponse: {result['answer'][:100]}...")
                return True
            else:
                print(f"❌ Échec requête: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur requête: {e}")
            return False
    
    def test_question_suggestions(self):
        """Test de génération de questions"""
        print("\n❓ Test de suggestions de questions...")
        try:
            payload = {
                "session_id": self.session_id,
                "context": "Dataset RH avec informations sur les employés",
                "num_questions": 4
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/questions/suggest",
                json=payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Questions générées avec succès")
                for i, q in enumerate(result['questions'][:3], 1):
                    print(f"  {i}. {q.get('question', q)}")
                return True
            else:
                print(f"❌ Échec questions: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur questions: {e}")
            return False
    
    def test_insights_generation(self):
        """Test de génération d'insights"""
        print("\n🔍 Test de génération d'insights...")
        try:
            payload = {
                "session_id": self.session_id,
                "analysis_type": "comprehensive"
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/insights/generate",
                json=payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Insights générés avec succès")
                for insight in result['insights'][:2]:
                    print(f"  💡 {insight.get('title', 'Insight')}")
                return True
            else:
                print(f"❌ Échec insights: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur insights: {e}")
            return False
    
    def test_chart_creation(self):
        """Test de création de graphique"""
        print("\n📊 Test de création de graphique...")
        try:
            payload = {
                "session_id": self.session_id,
                "query": "Graphique des salaires par département",
                "chart_type": "bar",
                "columns": ["departement", "salaire"]
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/charts/create",
                json=payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Graphique créé avec succès")
                print(f"📈 Type: {result['chart_type']}")
                print(f"📝 Description: {result['description']}")
                return True
            else:
                print(f"❌ Échec graphique: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur graphique: {e}")
            return False
    
    def test_session_management(self):
        """Test de gestion des sessions"""
        print("\n📋 Test de gestion des sessions...")
        try:
            # Lister les sessions
            response = requests.get(
                f"{self.base_url}/api/v1/sessions",
                headers=self.headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {len(result['sessions'])} sessions actives")
                return True
            else:
                print(f"❌ Échec sessions: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur sessions: {e}")
            return False
    
    def run_all_tests(self):
        """Exécute tous les tests"""
        print("🧪 SUITE DE TESTS API DATATALK")
        print("=" * 50)
        
        tests = [
            ("Santé", self.test_health),
            ("Upload", self.test_data_upload),
            ("Requête NLQ", self.test_chat_query),
            ("Questions", self.test_question_suggestions),
            ("Insights", self.test_insights_generation),
            ("Graphiques", self.test_chart_creation),
            ("Sessions", self.test_session_management)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"❌ Erreur critique dans {test_name}: {e}")
                results.append((test_name, False))
        
        # Résumé
        print("\n" + "=" * 50)
        print("📊 RÉSULTATS DES TESTS")
        success_count = 0
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} {test_name}")
            if success:
                success_count += 1
        
        print(f"\n🎯 Score: {success_count}/{len(results)} tests réussis")
        
        if success_count == len(results):
            print("🎉 Tous les tests sont PASSÉS! API fonctionnelle.")
        else:
            print("⚠️  Certains tests ont échoué. Vérifiez l'API.")

def main():
    """Point d'entrée principal"""
    print("🔧 Configuration du testeur API DataTalk...")
    
    # Vérifier si l'API est disponible
    tester = DataTalkAPITester(API_BASE_URL, API_KEY)
    
    if not tester.test_health():
        print("\n❌ L'API n'est pas accessible.")
        print("💡 Assurez-vous que l'API est démarrée avec: python run_api.py")
        return
    
    # Exécuter tous les tests
    tester.run_all_tests()

if __name__ == "__main__":
    main()