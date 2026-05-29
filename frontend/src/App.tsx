import { useState, useRef } from 'react';
import { UploadCloud, FileSpreadsheet, Clock, Loader2, Database, Activity, CheckCircle2, ChevronRight, Settings2, Zap, LayoutList, Target, Play } from 'lucide-react';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';

export default function App() {
  // --- App State ---
  const [appPhase, setAppPhase] = useState(1); // 1: Upload, 2: EDA, 3: Setup
  const [sessionData, setSessionData] = useState<any>(null);
  
  // --- Upload State ---
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- ML Configuration State (Phase 3) ---
  const [targetColumn, setTargetColumn] = useState('');
  const [engineMode, setEngineMode] = useState<'standard' | 'timeseries'>('standard');
  const [timestampColumn, setTimestampColumn] = useState('');
  const [selectedFeatures, setSelectedFeatures] = useState({
    extractDates: true,
    scaleNumeric: true,
    removeLowVariance: true,
  });
  const [selectedAlgos, setSelectedAlgos] = useState({
    xgboost: true,
    lightgbm: true,
    random_forest: true,
    svm: false,
    logistic: true,
  });

  // --- File Handling Functions ---
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); };
const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); 
    setIsDragging(false);
    
    // Explicitly check that files exists, then check length
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Explicitly check that files exists, then check length
    if (e.target.files && e.target.files.length > 0) {
      validateAndUpload(e.target.files[0]);
    }
    
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const validateAndUpload = async (file: File) => {
    if (!file.name.endsWith('.csv')) {
      toast.error('Please upload a valid .csv file');
      return;
    }
    await performUpload(file);
  };

  const performUpload = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => setUploadProgress(Math.round((e.loaded * 100) / (e.total ?? 1))),
      });
      toast.success('Dataset analyzed successfully!');
      setSessionData(response.data);
      setAppPhase(2); // Move to EDA Hub
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Upload failed');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  // --- Handlers for ML Config ---
  const toggleFeature = (key: keyof typeof selectedFeatures) => {
    setSelectedFeatures(prev => ({ ...prev, [key]: !prev[key] }));
  };
  
  const toggleAlgo = (key: keyof typeof selectedAlgos) => {
    setSelectedAlgos(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleRunAutoML = () => {
    if (!targetColumn) {
      toast.error("Please select a target column to predict.");
      return;
    }
    if (engineMode === 'timeseries' && !timestampColumn) {
      toast.error("Time-Series mode requires a timestamp column.");
      return;
    }
    
    console.log("Starting AutoML with config:", {
      targetColumn, engineMode, timestampColumn, selectedFeatures, selectedAlgos
    });
    // Next step: Send this to the backend!
    toast.success("Initializing AutoML Engine...");
  };


  // ==========================================
  // VIEW 3: COMMAND CENTER (Setup)
  // ==========================================
  if (appPhase === 3 && sessionData) {
    return (
      <div className="min-h-screen flex flex-col items-center pt-16 px-4 pb-24">
        <Toaster position="bottom-center" toastOptions={{ style: { background: '#1a1a1a', color: '#f5f5f5', border: '1px solid #2a2a2a' } }} />
        <div className="w-full max-w-4xl">
          
          <div className="flex items-center justify-between mb-8 pb-4 border-b border-border-subtle">
            <h2 className="text-2xl font-bold flex items-center gap-2">
              KyD<span className="text-accent-primary">.ai</span>
              <span className="text-text-muted font-normal text-xl mx-2">/</span>
              <span className="text-text-secondary font-medium text-xl">Setup</span>
            </h2>
            <div className="flex items-center gap-2 text-sm text-text-muted font-mono tracking-wider">
              1. EDA — <span className="text-accent-primary">2. SETUP</span> — 3. TRAIN — 4. EXPORT
            </div>
          </div>

          <div className="space-y-6">
            {/* 1. Target Column */}
            <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
              <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><Target size={18} className="text-accent-primary"/> 1. Select Target Column</h3>
              <p className="text-text-secondary text-sm mb-4">Choose the variable you want the ML model to predict.</p>
              <select 
                className="w-full p-3 rounded-lg bg-background-primary border border-border-subtle text-text-primary focus:border-accent-primary outline-none transition-colors"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
              >
                <option value="">-- Select a column --</option>
                {sessionData.column_names?.map((col: string) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>

            {/* 2. Engine Mode */}
            <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
              <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><Settings2 size={18} className="text-accent-primary"/> 2. Choose Engine Mode</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div 
                  onClick={() => setEngineMode('standard')}
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${engineMode === 'standard' ? 'border-accent-primary bg-accent-primary/5' : 'border-border-subtle hover:border-border-active'}`}
                >
                  <h4 className="font-medium mb-1">Standard Mode</h4>
                  <p className="text-sm text-text-secondary">Random train/test splits. Best for general classification and regression tasks.</p>
                </div>
                <div 
                  onClick={() => setEngineMode('timeseries')}
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${engineMode === 'timeseries' ? 'border-accent-primary bg-accent-primary/5' : 'border-border-subtle hover:border-border-active'}`}
                >
                  <h4 className="font-medium mb-1">Time-Series / Trading</h4>
                  <p className="text-sm text-text-secondary">Sequential temporal validation. Enforces walk-forward splits to prevent data leakage.</p>
                </div>
              </div>

              {engineMode === 'timeseries' && (
                <div className="mt-4 p-4 rounded-lg bg-background-primary border border-border-subtle animate-in fade-in slide-in-from-top-2">
                  <label className="block text-sm font-medium text-text-secondary mb-2">Select Timestamp Column</label>
                  <select 
                    className="w-full p-3 rounded-lg bg-background-surface border border-border-subtle text-text-primary focus:border-accent-primary outline-none"
                    value={timestampColumn}
                    onChange={(e) => setTimestampColumn(e.target.value)}
                  >
                    <option value="">-- Select timestamp column --</option>
                    {sessionData.column_names?.map((col: string) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {/* 3. Feature Engineering & Algorithms (Grid) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
                <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><Zap size={18} className="text-accent-primary"/> Preprocessing</h3>
                <div className="space-y-3">
                  {Object.entries(selectedFeatures).map(([key, value]) => (
                    <label key={key} className="flex items-center gap-3 cursor-pointer group">
                      <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${value ? 'bg-accent-primary border-accent-primary' : 'border-border-subtle group-hover:border-border-active'}`}>
                        {value && <CheckCircle2 size={14} className="text-white" />}
                      </div>
                      <span className="text-sm text-text-primary capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
                <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><LayoutList size={18} className="text-accent-primary"/> Algorithms</h3>
                <div className="space-y-3">
                  {Object.entries(selectedAlgos).map(([key, value]) => (
                    <label key={key} className="flex items-center gap-3 cursor-pointer group">
                      <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${value ? 'bg-accent-primary border-accent-primary' : 'border-border-subtle group-hover:border-border-active'}`}>
                        {value && <CheckCircle2 size={14} className="text-white" />}
                      </div>
                      <span className="text-sm text-text-primary capitalize">{key.replace('_', ' ')}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            {/* Action Bar */}
            <div className="flex justify-between items-center mt-8 pt-6 border-t border-border-subtle">
              <button onClick={() => setAppPhase(2)} className="px-6 py-3 text-text-secondary hover:text-text-primary transition-colors">
                Back to EDA
              </button>
              <button 
                onClick={handleRunAutoML}
                className="flex items-center gap-2 px-8 py-4 bg-accent-primary hover:bg-indigo-600 text-white rounded-lg font-medium transition-all shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)]"
              >
                <Play size={18} fill="currentColor" /> Run AutoML Engine
              </button>
            </div>
            
          </div>
        </div>
      </div>
    );
  }

  // ==========================================
  // VIEW 2: EDA HUB
  // ==========================================
  if (appPhase === 2 && sessionData) {
    return (
      <div className="min-h-screen flex flex-col items-center pt-16 px-4 pb-24">
        <Toaster position="bottom-center" toastOptions={{ style: { background: '#1a1a1a', color: '#f5f5f5', border: '1px solid #2a2a2a' } }} />
        <div className="w-full max-w-5xl">
          <div className="flex items-center justify-between mb-8 pb-4 border-b border-border-subtle">
            <h2 className="text-2xl font-bold flex items-center gap-2">
              KyD<span className="text-accent-primary">.ai</span>
              <span className="text-text-muted font-normal text-xl mx-2">/</span>
              <span className="text-text-secondary font-medium text-xl">{sessionData.filename}</span>
            </h2>
            <div className="flex items-center gap-2 text-sm text-text-muted font-mono tracking-wider">
              <span className="text-accent-primary">1. EDA</span> — 2. SETUP — 3. TRAIN — 4. EXPORT
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="p-6 rounded-xl border border-border-subtle bg-background-surface flex flex-col gap-1">
              <span className="text-text-muted text-sm font-medium flex items-center gap-2"><Database size={16}/> Rows</span>
              <span className="text-3xl font-bold font-mono">{sessionData.rows.toLocaleString()}</span>
            </div>
            <div className="p-6 rounded-xl border border-border-subtle bg-background-surface flex flex-col gap-1">
              <span className="text-text-muted text-sm font-medium flex items-center gap-2"><FileSpreadsheet size={16}/> Columns</span>
              <span className="text-3xl font-bold font-mono">{sessionData.columns}</span>
            </div>
            <div className="p-6 rounded-xl border border-border-subtle bg-background-surface flex flex-col gap-1">
              <span className="text-text-muted text-sm font-medium flex items-center gap-2"><Database size={16}/> Memory Size</span>
              <span className="text-3xl font-bold font-mono">{sessionData.memory_mb.toFixed(2)} MB</span>
            </div>
            <div className="p-6 rounded-xl border border-accent-success/30 bg-accent-success/5 flex flex-col gap-1 relative overflow-hidden">
              <span className="text-text-muted text-sm font-medium flex items-center gap-2 z-10"><Activity size={16}/> Health Score</span>
              <span className="text-3xl font-bold font-mono text-accent-success z-10">{sessionData.health_score}/100</span>
              <div className="absolute bottom-0 left-0 h-1 bg-accent-success" style={{ width: `${sessionData.health_score}%` }}></div>
            </div>
          </div>

          <div className="flex justify-end mt-8">
            <button onClick={() => setAppPhase(3)} className="flex items-center gap-2 px-8 py-4 bg-accent-primary hover:bg-indigo-600 text-white rounded-lg font-medium transition-colors shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)]">
              Proceed to ML Setup <ChevronRight size={20} />
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ==========================================
  // VIEW 1: UPLOAD ZONE (Default)
  // ==========================================
  return (
    <div className="min-h-screen flex flex-col items-center pt-24 px-4 relative">
      <Toaster position="bottom-center" toastOptions={{ style: { background: '#1a1a1a', color: '#f5f5f5', border: '1px solid #2a2a2a' } }} />
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4 text-text-primary">
          KyD<span className="text-accent-primary">.ai</span>
        </h1>
        <p className="text-text-secondary text-lg">Upload your dataset to get started.</p>
      </div>
      <div 
        className={`w-full max-w-2xl p-12 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all duration-200 ease-in-out bg-background-surface ${isDragging && !isUploading ? 'border-accent-primary bg-background-elevated' : ''} ${!isUploading ? 'border-border-subtle hover:border-border-active hover:bg-background-elevated cursor-pointer' : 'border-border-subtle opacity-80 cursor-not-allowed'}`}
        onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} onClick={() => !isUploading && fileInputRef.current?.click()}
      >
        <input type="file" accept=".csv" className="hidden" ref={fileInputRef} onChange={handleFileInput} />
        <div className={`p-4 rounded-full mb-4 ${isDragging && !isUploading ? 'bg-accent-primary/20 text-accent-primary' : 'bg-background-primary text-text-muted'}`}>
          {isUploading ? <Loader2 size={40} className="animate-spin text-accent-primary" /> : <UploadCloud size={40} />}
        </div>
        <h3 className="text-xl font-medium mb-2">{isUploading ? 'Uploading & Analyzing...' : 'Drag & Drop your CSV file'}</h3>
        <p className="text-text-muted mb-6">{isUploading ? 'Please wait while we crunch the numbers' : 'or click to browse files'}</p>
        {!isUploading && (
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <FileSpreadsheet size={16} /><span>Supports: .csv • Max size: 500MB</span>
          </div>
        )}
        {isUploading && (
          <div className="w-full max-w-md mt-6">
            <div className="h-2 w-full bg-background-primary rounded-full overflow-hidden border border-border-subtle">
              <div className="h-full bg-accent-primary transition-all duration-300 ease-out" style={{ width: `${uploadProgress}%` }} />
            </div>
            <p className="text-center text-sm text-text-secondary mt-2">{uploadProgress}%</p>
          </div>
        )}
      </div>
    </div>
  );
}