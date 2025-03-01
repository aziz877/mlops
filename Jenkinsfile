pipeline {
    agent any
    
    environment {
        VENV_DIR = 'venv'
        TRAIN_DATA = 'churn-bigml-80.csv'
        TEST_DATA = 'churn-bigml-20.csv'
        MODEL_FILE = 'modelRF.joblib'
    }
    
    stages {
        stage('Setup Environment') {
            steps {
                script {
                    if (!fileExists(VENV_DIR)) {
                        sh 'python3 -m venv venv'
                    }
                    sh 'source venv/bin/activate && pip install -r requirements.txt'
                }
            }
        }
        
        stage('Prepare Data') {
            steps {
                sh 'source venv/bin/activate && python main.py prepare $TRAIN_DATA $TEST_DATA'
            }
        }
        
        stage('Train Model') {
            steps {
                sh 'source venv/bin/activate && python main.py train $TRAIN_DATA $TEST_DATA --model_filename=$MODEL_FILE'
            }
        }
        
        stage('Evaluate Model') {
            steps {
                sh 'source venv/bin/activate && python main.py evaluate $TRAIN_DATA $TEST_DATA --model_filename=$MODEL_FILE'
            }
        }
        
        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'modelRF.joblib, train.log, roc_curve.png', fingerprint: true
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for details.'
        }
    }
}

