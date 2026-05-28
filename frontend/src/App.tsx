import { useState, useRef } from 'react';
import { UploadCloud, FileSpreadsheet, Clock } from 'lucide-react';

export default function App() {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  // 1. Create a reference to our hidden file input
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
      validateAndSetFile(file);
    }
  };

  // 2. Handle the file when selected via click
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      validateAndSetFile(file);
    }
  };

  // 3. Helper function to validate the file
  const validateAndSetFile = (file: File) => {
    if (file.name.endsWith('.csv')) {
      setSelectedFile(file);
      console.log("File captured in state:", file);
    } else {
      alert('Please upload a valid .csv file');
    }
  };

  // 4. Trigger the hidden input when the box is clicked
  const handleBoxClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="min-h-screen flex flex-col items-center pt-24 px-4">
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
        className={`w-full max-w-2xl p-12 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all duration-200 ease-in-out cursor-pointer bg-background-surface
          ${isDragging 
            ? 'border-accent-primary bg-background-elevated' 
            : 'border-border-subtle hover:border-border-active hover:bg-background-elevated'
          }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleBoxClick}
      >
        {/* Hidden File Input */}
        <input 
          type="file" 
          accept=".csv" 
          className="hidden" 
          ref={fileInputRef}
          onChange={handleFileInput}
        />

        <div className={`p-4 rounded-full mb-4 ${isDragging ? 'bg-accent-primary/20 text-accent-primary' : 'bg-background-primary text-text-muted'}`}>
          <UploadCloud size={40} />
        </div>
        
        <h3 className="text-xl font-medium mb-2">
          {isDragging ? 'Drop CSV here...' : 'Drag & Drop your CSV file'}
        </h3>
        <p className="text-text-muted mb-6">or click to browse files</p>
        
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <FileSpreadsheet size={16} />
          <span>Supports: .csv • Max size: 500MB</span>
        </div>

        {selectedFile && (
          <div className="mt-6 p-3 bg-background-primary rounded-lg border border-border-subtle text-accent-success flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
            <FileSpreadsheet size={16} />
            {selectedFile.name} ready for upload.
          </div>
        )}
      </div>

      {/* Recent Sessions Placeholder */}
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