import { useState, useRef } from 'react';
import { UploadCloud, FileSpreadsheet, Clock, Loader2 } from 'lucide-react';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';

export default function App() {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      validateAndUpload(file);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      validateAndUpload(file);
    }
    // Reset input so the same file can be selected again if needed
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const validateAndUpload = async (file: File) => {
    if (!file.name.endsWith('.csv')) {
      toast.error('Please upload a valid .csv file');
      return;
    }
    
    setSelectedFile(file);
    await performUpload(file);
  };

  // The core API call to our FastAPI backend
  const performUpload = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total ?? 1)
          );
          setUploadProgress(percentCompleted);
        },
      });

      toast.success('Dataset analyzed successfully!');
      console.log("FastAPI Response:", response.data);
      
      // We will add the routing to Phase 1 (EDA Hub) here later!

    } catch (error: any) {
      console.error("Upload failed:", error);
      toast.error(error.response?.data?.detail || 'Failed to connect to backend.');
      setSelectedFile(null);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleBoxClick = () => {
    if (!isUploading) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center pt-24 px-4 relative">
      <Toaster 
        position="bottom-center"
        toastOptions={{
          style: { background: '#1a1a1a', color: '#f5f5f5', border: '1px solid #2a2a2a' }
        }} 
      />

      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4 text-text-primary">
          KyD<span className="text-accent-primary">.ai</span>
        </h1>
        <p className="text-text-secondary text-lg">
          Upload your dataset to get started.
        </p>
      </div>

      {/* Upload Zone */}
      <div 
        className={`w-full max-w-2xl p-12 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all duration-200 ease-in-out bg-background-surface
          ${isDragging && !isUploading ? 'border-accent-primary bg-background-elevated' : ''}
          ${!isUploading ? 'border-border-subtle hover:border-border-active hover:bg-background-elevated cursor-pointer' : 'border-border-subtle opacity-80 cursor-not-allowed'}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleBoxClick}
      >
        <input 
          type="file" 
          accept=".csv" 
          className="hidden" 
          ref={fileInputRef}
          onChange={handleFileInput}
        />

        <div className={`p-4 rounded-full mb-4 ${isDragging && !isUploading ? 'bg-accent-primary/20 text-accent-primary' : 'bg-background-primary text-text-muted'}`}>
          {isUploading ? <Loader2 size={40} className="animate-spin text-accent-primary" /> : <UploadCloud size={40} />}
        </div>
        
        <h3 className="text-xl font-medium mb-2">
          {isUploading ? 'Uploading & Analyzing...' : isDragging ? 'Drop CSV here...' : 'Drag & Drop your CSV file'}
        </h3>
        <p className="text-text-muted mb-6">
          {isUploading ? 'Please wait while we crunch the numbers' : 'or click to browse files'}
        </p>
        
        {!isUploading && (
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <FileSpreadsheet size={16} />
            <span>Supports: .csv • Max size: 500MB</span>
          </div>
        )}

        {/* Progress Bar */}
        {isUploading && (
          <div className="w-full max-w-md mt-6">
            <div className="h-2 w-full bg-background-primary rounded-full overflow-hidden border border-border-subtle">
              <div 
                className="h-full bg-accent-primary transition-all duration-300 ease-out"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p className="text-center text-sm text-text-secondary mt-2">{uploadProgress}%</p>
          </div>
        )}
      </div>

      {/* Recent Sessions */}
      <div className="w-full max-w-2xl mt-12">
        <h4 className="text-text-secondary font-medium mb-4 flex items-center gap-2">
          <Clock size={16} /> Recent Sessions
        </h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 rounded-lg border border-border-subtle bg-background-surface hover:border-border-active cursor-pointer transition-colors">
            <p className="font-medium truncate">customer_churn_data.csv</p>
            <p className="text-sm text-text-muted mt-1">2 hours ago</p>
          </div>
          <div className="p-4 rounded-lg border border-border-subtle bg-background-surface hover:border-border-active cursor-pointer transition-colors">
            <p className="font-medium truncate">q1_financials_2026.csv</p>
            <p className="text-sm text-text-muted mt-1">Yesterday</p>
          </div>
        </div>
      </div>
    </div>
  );
}