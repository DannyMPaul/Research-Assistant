import React, { useState } from 'react';
import DocumentUpload from './components/DocumentUpload';
import DocumentList from './components/DocumentList';

function App() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadSuccess = (result) => {
    console.log('Upload successful:', result);
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Document Research Assistant
          </h1>
          <p className="text-gray-600">
            Upload documents and extract insights
          </p>
          <div className="text-sm text-gray-500 mt-2">
            Version 0.1.0 - Base Infrastructure
          </div>
        </header>

        <div className="space-y-8">
          <DocumentUpload onUploadSuccess={handleUploadSuccess} />
          <DocumentList refreshTrigger={refreshTrigger} />
        </div>
      </div>
    </div>
  );
}

export default App;
