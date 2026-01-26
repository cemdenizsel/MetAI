// MongoDB initialization script
// Create the required database and user with proper permissions

// Switch to admin database to create user
db = db.getSiblingDB('admin');

// Create a user with root privileges to manage the database
db.createUser({
  user: 'admin',
  pwd: 'password',
  roles: [
    { role: 'root', db: 'admin' }
  ]
});

// Switch to the application database
db = db.getSiblingDB('emotion_analysis_db');

// Create a user with readWrite permissions for the application
db.createUser({
  user: 'emotion_user',
  pwd: 'emotion_password',
  roles: [
    { role: 'readWrite', db: 'emotion_analysis_db' },
    { role: 'dbAdmin', db: 'emotion_analysis_db' }
  ]
});

// Create collections and indexes if needed
db.createCollection('chat_sessions');
db.createCollection('users');

// Create indexes
db.chat_sessions.createIndex({ "session_id": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true });

print('MongoDB initialization completed');