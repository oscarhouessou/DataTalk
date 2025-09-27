#!/usr/bin/env python3
"""
Test simple de l'API DataTalk
Vérifie les endpoints principaux
"""

import requests
import pandas as pd
import io

# Configuration
API_BASE_URL = "http://localhost:8000"

def create_test_data():
    """Crée un dataset de test"""
    data = {
        'nom': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salaire': [50000, 60000, 70000, 55000, 65000],
        'departement': ['IT', 'Marketing', 'IT', 'RH', 'Marketing']
    }
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def test_health():
    """Test de l'endpoint de santé"""
    print("🔍 Test de santé de l'API...")
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API en bonne santé - OpenAI configuré: {data['openai_configured']}")
        return True
    else:
        print(f"❌ Problème de santé: {response.status_code}")
        return False

def test_upload():
    """Test d'upload de données"""
    print("\n📤 Test d'upload de données...")
    csv_data = create_test_data()
    
    files = {'file': ('test_data.csv', csv_data, 'text/csv')}
    data = {'session_id': 'test-session-123'}
    
    response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Upload réussi: {result['rows']} lignes, {result['columns']} colonnes")
        print(f"📊 Colonnes: {', '.join(result['column_names'])}")
        return True
    else:
        print(f"❌ Échec upload: {response.status_code} - {response.text}")
        return False

def test_chat():
    """Test de requête en chat"""
    print("\n💬 Test de requête chat...")
    payload = {
        "session_id": "test-session-123",
        "query": "Quel est le salaire moyen par département?"
    }
    
    response = requests.post(f"{API_BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Requête traitée avec succès")
        print(f"🤖 Réponse: {result['answer'][:150]}...")
        return True
    else:
        print(f"❌ Échec chat: {response.status_code} - {response.text}")
        return False

def test_questions():
    """Test de génération de questions"""
    print("\n❓ Test de suggestions de questions...")
    payload = {"session_id": "test-session-123"}
    
    response = requests.post(f"{API_BASE_URL}/questions", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Questions générées avec succès")
        for i, q in enumerate(result['questions'][:3], 1):
            print(f"  {i}. {q}")
        return True
    else:
        print(f"❌ Échec questions: {response.status_code} - {response.text}")
        return False

def test_sessions():
    """Test de gestion des sessions"""
    print("\n📋 Test de gestion des sessions...")
    
    response = requests.get(f"{API_BASE_URL}/sessions")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {len(result['sessions'])} sessions actives")
        for session in result['sessions']:
            print(f"  📊 {session['session_id']}: {session['rows']} lignes")
        return True
    else:
        print(f"❌ Échec sessions: {response.status_code} - {response.text}")
        return False

def main():
    """Exécute tous les tests"""
    print("🧪 TESTS DE L'API DATATALK")
    print("=" * 40)
    
    tests = [
        ("Santé", test_health),
        ("Upload", test_upload),
        ("Chat", test_chat),
        ("Questions", test_questions),
        ("Sessions", test_sessions)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Erreur dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé
    print("\n" + "=" * 40)
    print("📊 RÉSULTATS")
    success_count = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
        if success:
            success_count += 1
    
    print(f"\n🎯 Score: {success_count}/{len(results)} tests réussis")
    
    if success_count == len(results):
        print("🎉 Tous les tests PASSENT! API fonctionnelle.")
    else:
        print("⚠️  Certains tests ont échoué.")

if __name__ == "__main__":
    main()