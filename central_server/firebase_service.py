import firebase_admin
from firebase_admin import credentials, db
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class FirebaseService:
    def __init__(self):
        try:
            # Initialize Firebase Admin SDK
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": os.getenv('FIREBASE_PROJECT_ID'),
                "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
                "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
                "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
                "client_id": os.getenv('FIREBASE_CLIENT_ID'),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL')
            })
            
            # Initialize with direct database URL
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://traffic-61ac0-default-rtdb.asia-southeast1.firebasedatabase.app/'
            })
            
            # Get a reference to the database
            self.db = db.reference('/')
            
            # Initialize data directory
            self.data_dir = os.path.join(os.path.dirname(__file__), 'server_data')
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Initialize the database structure
            self._initialize_database()
            
        except Exception as e:
            print(f"\n❌ Error initializing Firebase: {str(e)}")
            print("\nPlease verify:")
            print("1. Your Firebase project exists and has Realtime Database enabled")
            print("2. Your service account has the necessary permissions")
            raise e

    def _initialize_database(self):
        """Initialize the database structure"""
        try:
            print("\nInitializing database structure...")
            print(f"Data directory: {self.data_dir}")
            
            # Initialize status (local only)
            status_data = {
                'total_agents': 0,
                'online_agents': 0,
                'last_updated': datetime.now().isoformat()
            }
            self._save_to_json('status.json', status_data)
            print("✓ Status node initialized")

            # Initialize agents node
            agents_data = {}
            self._save_to_json('agent_data.json', agents_data)
            self._upload_json_to_rtdb('agent_data.json')
            print("✓ Agents node initialized")
            
            print("Database structure initialized successfully\n")
            
        except Exception as e:
            print(f"❌ Error initializing database structure: {str(e)}")
            raise e

    def _save_to_json(self, filename, data):
        """Save data to JSON file"""
        try:
            filepath = os.path.join(self.data_dir, filename)
            print(f"\nSaving to file: {filepath}")
            print(f"Data to save: {json.dumps(data, indent=2)}")
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Successfully saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving to JSON file {filename}: {str(e)}")
            print(f"File path: {filepath}")
            print(f"Data that failed to save: {json.dumps(data, indent=2)}")
            raise e

    def _load_from_json(self, filename):
        """Load data from JSON file"""
        try:
            filepath = os.path.join(self.data_dir, filename)
            print(f"\nLoading from file: {filepath}")
            if os.path.exists(filepath):
                print(f"File exists, size: {os.path.getsize(filepath)} bytes")
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"✓ Successfully loaded from {filename}")
                json_str = json.dumps(data, indent=2)
                print(f"Loaded data: {json_str[:30]}...")
                return data
            else:
                print(f"❌ File does not exist: {filepath}")
                return {}
        except Exception as e:
            print(f"❌ Error loading from JSON file {filename}: {str(e)}")
            print(f"File path: {filepath}")
            if os.path.exists(filepath):
                print(f"File exists but failed to read. Size: {os.path.getsize(filepath)} bytes")
                try:
                    with open(filepath, 'r') as f:
                        print(f"Raw file contents: {f.read()}")
                except Exception as read_error:
                    print(f"Could not read raw file contents: {str(read_error)}")
            return {}

    def _upload_json_to_rtdb(self, filename):
        """Upload entire JSON file to RTDB"""
        try:
            print(f"\n=== Starting RTDB Upload for {filename} ===")
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"❌ File does not exist: {filepath}")
                return
                
            # Read the JSON file
            with open(filepath, 'r') as file:
                data = json.load(file)
            
            # Upload the entire JSON data
            if filename == 'agent_data.json':
                print("Uploading agent data to RTDB...")
                self.db.child('agents').set(data)
                print("✓ Agent data uploaded successfully to RTDB")
            elif filename == 'status.json':
                print("Skipping status.json upload as it's local-only")
            else:
                print(f"Warning: Unknown file type {filename}")
                
            print("=== RTDB Upload Complete ===\n")
            
        except Exception as e:
            print(f"❌ Error uploading {filename} to RTDB: {str(e)}")
            print("=== RTDB Upload Failed ===\n")
            raise e

    def test_database_connection(self):
        """Test the database connection by writing and reading test data"""
        try:
            print("\n=== Testing Firebase RTDB Connection ===")
            
            # Test writing to a test node
            test_data = {
                'test_timestamp': datetime.now().isoformat(),
                'test_message': 'Hello from Python!',
                'test_number': 42
            }
            
            print("Writing test data...")
            self.db.child('test').set(test_data)
            print("✓ Test data written successfully")
            
            # Test reading the test data
            print("Reading test data...")
            read_data = self.db.child('test').get()
            print(f"✓ Test data read successfully: {read_data}")
            
            # Test updating the test data
            print("Updating test data...")
            self.db.child('test').update({
                'test_message': 'Updated message!',
                'last_updated': datetime.now().isoformat()
            })
            print("✓ Test data updated successfully")
            
            # Verify the update
            updated_data = self.db.child('test').get()
            print(f"✓ Updated data: {updated_data}")
            
            print("=== Firebase RTDB Connection Test Complete ===\n")
            
        except Exception as e:
            print(f"\n❌ Error testing database connection: {e}\n")
            raise e

    def update_agent_status(self, agent_id, status):
        """Update agent status in both RTDB and JSON"""
        try:
            print(f"\nUpdating status for agent {agent_id}...")
            # Update JSON file
            agents_data = self._load_from_json('agent_data.json')
            print(f"Current agent data: {json.dumps(agents_data, indent=2)[:30]}...")
            
            if agent_id not in agents_data:
                agents_data[agent_id] = {}
            
            # Update status and set online based on status
            agents_data[agent_id].update({
                'status': status,
                'online': status != 'terminated',  # Set online based on status
                'last_updated': datetime.now().isoformat()
            })
            print(f"Updated agent data: {json.dumps(agents_data, indent=2)[:30]}...")
            
            self._save_to_json('agent_data.json', agents_data)
            print("✓ Saved agent data to JSON file")
            
            # Upload entire JSON to RTDB
            print("\n=== Starting RTDB Upload ===")
            print("Uploading agent data to RTDB...")
            print(f"Uploading to path: /agents")
            self.db.child('agents').set(agents_data)
            print("✓ Agent data uploaded successfully to RTDB")
            print("=== RTDB Upload Complete ===\n")
            
            print(f"✓ Completed status update for agent {agent_id}")
        except Exception as e:
            print(f"❌ Error updating agent status: {str(e)}")
            raise e

    def update_agent_metrics(self, agent_id, metrics):
        """Update agent metrics in both RTDB and JSON"""
        try:
            print(f"\nUpdating metrics for agent {agent_id}...")
            # Update JSON file
            agents_data = self._load_from_json('agent_data.json')
            print(f"Current agent data: {json.dumps(agents_data, indent=2)[:30]}...")
            
            if agent_id not in agents_data:
                agents_data[agent_id] = {}
            if 'metrics' not in agents_data[agent_id]:
                agents_data[agent_id]['metrics'] = {}
            agents_data[agent_id]['metrics'].update(metrics)
            print(f"Updated agent data: {json.dumps(agents_data, indent=2)[:30]}...")
            
            self._save_to_json('agent_data.json', agents_data)
            print("✓ Saved agent data to JSON file")
            
            # Upload entire JSON to RTDB
            print("\n=== Starting RTDB Upload ===")
            print("Uploading agent data to RTDB...")
            print(f"Uploading to path: /agents")
            self.db.child('agents').set(agents_data)
            print("✓ Agent data uploaded successfully to RTDB")
            print("=== RTDB Upload Complete ===\n")
            
            print(f"✓ Completed metrics update for agent {agent_id}")
        except Exception as e:
            print(f"❌ Error updating agent metrics: {str(e)}")
            raise e

    def update_agent_config(self, agent_id, config):
        """Update agent configuration in both RTDB and JSON"""
        try:
            print(f"\nUpdating config for agent {agent_id}...")
            # Update JSON file
            agents_data = self._load_from_json('agent_data.json')
            print(f"Current agent data: {json.dumps(agents_data, indent=2)}")
            
            if agent_id not in agents_data:
                agents_data[agent_id] = {}
            if 'config' not in agents_data[agent_id]:
                agents_data[agent_id]['config'] = {}
            agents_data[agent_id]['config'].update(config)
            print(f"Updated agent data: {json.dumps(agents_data, indent=2)}")
            
            self._save_to_json('agent_data.json', agents_data)
            print("✓ Saved agent data to JSON file")
            
            # Upload entire JSON to RTDB
            self._upload_json_to_rtdb('agent_data.json')
            
            print(f"✓ Completed config update for agent {agent_id}")
        except Exception as e:
            print(f"❌ Error updating agent config: {str(e)}")
            raise e

    def add_agent_log(self, agent_id, message, level='INFO'):
        """Add a log entry for an agent in both RTDB and JSON"""
        try:
            print(f"\nAdding log for agent {agent_id}...")
            # Update JSON file
            agents_data = self._load_from_json('agent_data.json')
            print(f"Current agent data: {json.dumps(agents_data, indent=2)}")
            
            if agent_id not in agents_data:
                agents_data[agent_id] = {}
            if 'logs' not in agents_data[agent_id]:
                agents_data[agent_id]['logs'] = []
            
            log_entry = {
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat()
            }
            agents_data[agent_id]['logs'].append(log_entry)
            print(f"Updated agent data: {json.dumps(agents_data, indent=2)}")
            
            self._save_to_json('agent_data.json', agents_data)
            print("✓ Saved agent data to JSON file")
            
            # Upload entire JSON to RTDB
            self._upload_json_to_rtdb('agent_data.json')
            
            print(f"✓ Completed log addition for agent {agent_id}")
        except Exception as e:
            print(f"❌ Error adding agent log: {str(e)}")
            raise e

    def update_agent_location(self, agent_id, latitude, longitude):
        """Update agent location in both RTDB and JSON"""
        try:
            print(f"\nUpdating location for agent {agent_id}...")
            # Update JSON file
            agents_data = self._load_from_json('agent_data.json')
            print(f"Current agent data: {json.dumps(agents_data, indent=2)}")
            
            if agent_id not in agents_data:
                agents_data[agent_id] = {}
            
            agents_data[agent_id].update({
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'last_updated': datetime.now().isoformat()
            })
            print(f"Updated agent data: {json.dumps(agents_data, indent=2)}")
            
            self._save_to_json('agent_data.json', agents_data)
            print("✓ Saved agent data to JSON file")
            
            # Upload entire JSON to RTDB
            self._upload_json_to_rtdb('agent_data.json')
            
            print(f"✓ Completed location update for agent {agent_id}")
        except Exception as e:
            print(f"❌ Error updating agent location: {str(e)}")
            raise e

    def update_agent_performance(self, agent_id, performance_data):
        """Update agent performance metrics in both RTDB and JSON"""
        try:
            print(f"\nUpdating performance for agent {agent_id}...")
            # Update JSON file
            agents_data = self._load_from_json('agent_data.json')
            print(f"Current agent data: {json.dumps(agents_data, indent=2)[:30]}...")
            
            if agent_id not in agents_data:
                agents_data[agent_id] = {}
            if 'performance' not in agents_data[agent_id]:
                agents_data[agent_id]['performance'] = {}
            
            agents_data[agent_id]['performance'].update({
                'latest_reward': performance_data.get('reward', 0),
                'queue_length': performance_data.get('queue_length', 0),
                'waiting_time': performance_data.get('waiting_time', 0),
                'last_updated': datetime.now().isoformat()
            })
            print(f"Updated agent data: {json.dumps(agents_data, indent=2)[:30]}...")
            
            self._save_to_json('agent_data.json', agents_data)
            print("✓ Saved agent data to JSON file")
            
            # Upload entire JSON to RTDB
            print("\n=== Starting RTDB Upload ===")
            print("Uploading agent data to RTDB...")
            print(f"Uploading to path: /agents")
            self.db.child('agents').set(agents_data)
            print("✓ Agent data uploaded successfully to RTDB")
            print("=== RTDB Upload Complete ===\n")
            
            print(f"✓ Completed performance update for agent {agent_id}")
        except Exception as e:
            print(f"❌ Error updating agent performance: {str(e)}")
            raise e

    def update_system_status(self, total_agents, online_agents):
        """Update overall system status (local only)"""
        try:
            # Get current agent data to count online agents
            agents_data = self._load_from_json('agent_data.json')
            online_count = sum(1 for agent in agents_data.values() if agent.get('online', False))
            
            # Update JSON file
            status_data = {
                'total_agents': total_agents,
                'online_agents': online_count,  # Use actual count from agent data
                'last_updated': datetime.now().isoformat()
            }
            self._save_to_json('status.json', status_data)
            print(f"Updated system status: {total_agents} total, {online_count} online")
        except Exception as e:
            print(f"❌ Error updating system status: {str(e)}")
            raise e

    def get_agent_data(self, agent_id):
        """Get agent data from JSON file (faster than RTDB)"""
        try:
            agents_data = self._load_from_json('agent_data.json')
            return agents_data.get(agent_id, {})
        except Exception as e:
            print(f"Error getting agent data: {str(e)}")
            return {}

    def get_all_agents(self):
        """Get all agents data from JSON file (faster than RTDB)"""
        try:
            return self._load_from_json('agent_data.json')
        except Exception as e:
            print(f"Error getting all agents: {str(e)}")
            return {}

    def get_system_status(self):
        """Get system status from JSON file (faster than RTDB)"""
        try:
            return self._load_from_json('status.json')
        except Exception as e:
            print(f"Error getting system status: {str(e)}")
            return {
                'total_agents': 0,
                'online_agents': 0,
                'last_updated': datetime.now().isoformat()
            }

    def delete_agent(self, agent_id):
        """Delete an agent from both RTDB and JSON"""
        try:
            # Update JSON file
            agents_data = self._load_from_json('agent_data.json')
            if agent_id in agents_data:
                del agents_data[agent_id]
                self._save_to_json('agent_data.json', agents_data)
                
                # Upload entire JSON to RTDB
                self._upload_json_to_rtdb('agent_data.json')
            
            print(f"Deleted agent {agent_id}")
        except Exception as e:
            print(f"Error deleting agent: {str(e)}")

    def clear_agent_logs(self, agent_id):
        """Clear all logs for an agent in both RTDB and JSON"""
        try:
            # Update JSON file
            agents_data = self._load_from_json('agent_data.json')
            if agent_id in agents_data and 'logs' in agents_data[agent_id]:
                agents_data[agent_id]['logs'] = []
                self._save_to_json('agent_data.json', agents_data)
                
                # Upload entire JSON to RTDB
                self._upload_json_to_rtdb('agent_data.json')
            
            print(f"Cleared logs for agent {agent_id}")
        except Exception as e:
            print(f"Error clearing agent logs: {str(e)}") 