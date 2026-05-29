import { useState, useRef, useEffect } from 'react';
import { 
  UploadCloud, FileSpreadsheet, Clock, Loader2, Database, 
  Activity, CheckCircle2, ChevronRight, Settings2, Zap, 
  LayoutList, Target, Play, Trophy, BarChart3, Timer, 
  Download, AlertTriangle, Network, SearchX, Info, Lightbulb, SplitSquareVertical, Filter, Terminal, FileCode2
} from 'lucide-react';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import { BarChart, Bar, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';

export default function App() {
  // --- App State ---
  const [appPhase, setAppPhase] = useState(1);
  const [sessionData, setSessionData] = useState<any>(null);
  const [leaderboardData, setLeaderboardData] = useState<any>(null);
  const [notebookUrl, setNotebookUrl] = useState<string>('');
  
  // --- Upload & UI State ---
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- Dynamic Training Console State ---
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // --- ML Configuration State (Phase 3) ---
  const [targetColumn, setTargetColumn] = useState('');
  const [engineMode, setEngineMode] = useState<'standard' | 'timeseries'>('standard');
  const [timestampColumn, setTimestampColumn] = useState('');
  const [imputationStrategy, setImputationStrategy] = useState<Record<string, string>>({});
  const [droppedColumns, setDroppedColumns] = useState<string[]>([]);
  
  // --- Bivariate State ---
  const [edaTarget, setEdaTarget] = useState('');
  const [bivariateData, setBivariateData] = useState<any>(null);
  const [isFetchingBivariate, setIsFetchingBivariate] = useState(false);

  const [selectedFeatures, setSelectedFeatures] = useState({ 
    extractDates: true, scaleNumeric: true, removeLowVariance: true 
  });
  const [selectedAlgos, setSelectedAlgos] = useState({ 
    xgboost: true, lightgbm: true, random_forest: true, svm: false, logistic: true 
  });

  // Auto-scroll terminal when logs update
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [trainingLogs]);

  // ==========================================
  // HANDLERS
  // ==========================================
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) validateAndUpload(e.dataTransfer.files[0]);
  };
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) validateAndUpload(e.target.files[0]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const validateAndUpload = async (file: File) => {
    if (!file.name.endsWith('.csv')) return toast.error('Please upload a valid .csv file');
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
      
      const initialImputations: Record<string, string> = {};
      response.data.eda.missing_summary.forEach((col: any) => {
        initialImputations[col.column] = col.dtype.includes('float') || col.dtype.includes('int') ? 'median' : 'mode';
      });
      setImputationStrategy(initialImputations);
      
      setAppPhase(2); 
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Upload failed');
    } finally {
      setIsUploading(false); 
      setUploadProgress(0);
    }
  };

  const handleEdaTargetSelect = async (selectedTarget: string) => {
    setEdaTarget(selectedTarget); 
    setTargetColumn(selectedTarget);
    
    if (!selectedTarget) {
      setBivariateData(null);
      return; 
    }
    
    setIsFetchingBivariate(true);
    try {
      const response = await axios.post('http://localhost:8000/api/bivariate', { 
        dataset_id: sessionData.dataset_id, 
        target_column: selectedTarget 
      });
      
      if (response.data.status === 'success') { 
        setBivariateData(response.data); 
        toast.success("Bivariate analysis generated!"); 
      } else { 
        toast.error(response.data.message); 
        setBivariateData(null); 
      }
    } catch (error: any) { 
      toast.error('Failed to generate bivariate relationships'); 
    } finally { 
      setIsFetchingBivariate(false); 
    }
  };

  const toggleFeature = (key: keyof typeof selectedFeatures) => setSelectedFeatures(prev => ({ ...prev, [key]: !prev[key] }));
  const toggleAlgo = (key: keyof typeof selectedAlgos) => setSelectedAlgos(prev => ({ ...prev, [key]: !prev[key] }));
  const handleImputationChange = (colName: string, strategy: string) => setImputationStrategy(prev => ({ ...prev, [colName]: strategy }));
  const toggleDropColumn = (col: string) => setDroppedColumns(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]);

  const handleRunAutoML = async () => {
    if (!targetColumn) return toast.error("Please select a target column to predict.");
    if (engineMode === 'timeseries' && !timestampColumn) return toast.error("Time-Series mode requires a timestamp column.");
    
    setIsTraining(true);
    setTrainingProgress(5);
    setTrainingLogs(['[SYSTEM] Initializing KyD.ai Engine...']);

    // --- DYNAMIC FAKE CONSOLE STREAMER ---
    const steps = [
      `[DATA] Validating target column: ${targetColumn}`,
      `[PREP] Dropping columns: ${droppedColumns.length > 0 ? droppedColumns.join(', ') : 'None'}`,
      `[PREP] Applying imputation strategies to missing data...`,
      `[SPLIT] Performing 80/20 train-test split...`,
      `[PREP] Encoding categorical text strings...`
    ];

    Object.entries(selectedAlgos).forEach(([key, isSelected]) => {
      if (isSelected) steps.push(`[ML] Training ${key.replace('_', ' ').toUpperCase()} model...`);
    });

    steps.push('[EVAL] Computing cross-validation accuracy & F1 scores...');
    steps.push('[EXPORT] Serializing .joblib model weights...');
    steps.push('[EXPORT] Generating Jupyter Notebook artifacts...');

    let currentStep = 0;
    const logInterval = setInterval(() => {
      if (currentStep < steps.length) {
        setTrainingLogs(prev => [...prev, steps[currentStep]]);
        setTrainingProgress(prev => Math.min(95, prev + (90 / steps.length)));
        currentStep++;
      } else {
        setTrainingLogs(prev => [...prev, '[SYSTEM] Optimizing final hyperparameters (Please wait)...']);
        clearInterval(logInterval);
      }
    }, 1200);

    try {
      const response = await axios.post('http://localhost:8000/api/train', {
        dataset_id: sessionData.dataset_id, 
        target_column: targetColumn, 
        engine_mode: engineMode,
        features: selectedFeatures, 
        algos: selectedAlgos, 
        drop_columns: droppedColumns, 
        imputation_strategy: imputationStrategy 
      });
      
      clearInterval(logInterval);
      setTrainingLogs(prev => [...prev, '[SUCCESS] Pipeline Complete! Redirecting to dashboard...']);
      setTrainingProgress(100);
      
      // Add a slight delay so the user can see the 100% completion
      setTimeout(() => {
        setLeaderboardData(response.data.leaderboard);
        setNotebookUrl(response.data.notebook_download_url);
        setAppPhase(4); 
        setIsTraining(false);
        setTrainingLogs([]);
      }, 800);

    } catch (error: any) { 
      clearInterval(logInterval);
      setTrainingLogs(prev => [...prev, `[ERROR] ${error.response?.data?.detail || 'Training failed'}`]);
      toast.error('Training failed. See console.');
      setTimeout(() => setIsTraining(false), 3000);
    } 
  };

  const getCorrelationColor = (val: number) => {
    if (val === 1) return 'bg-background-elevated text-text-muted'; 
    if (val > 0) return `bg-indigo-500 text-white font-medium`; 
    if (val < 0) return `bg-red-500 text-white font-medium`; 
    return 'bg-background-surface text-text-primary';
  };
  
  const getCorrelationOpacity = (val: number) => Math.max(0.15, Math.abs(val)); 
  const CHART_COLORS = ['#ef4444', '#22c55e', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899'];

  // ==========================================
  // COMPONENT: NAVIGATION HEADER
  // ==========================================
  const HeaderNav = () => (
    <div className="flex items-center justify-between mb-8 pb-4 border-b border-border-subtle w-full">
      <h2 
        onClick={() => setAppPhase(1)} 
        className="text-2xl font-bold tracking-tight flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
      >
        KyD<span className="text-accent-primary">.ai</span>
        {sessionData && <span className="text-text-muted font-normal text-xl mx-2">/</span>}
        {sessionData && <span className="text-text-secondary font-medium text-xl">{sessionData.filename}</span>}
      </h2>
      <div className="flex items-center gap-3 text-sm text-text-muted font-mono tracking-wider">
        <button 
          onClick={() => sessionData && setAppPhase(2)} 
          className={`transition-colors font-semibold ${appPhase === 2 ? 'text-accent-primary' : 'hover:text-white'}`}
        >
          1. EDA
        </button> 
        —
        <button 
          onClick={() => sessionData && setAppPhase(3)} 
          className={`transition-colors font-semibold ${appPhase === 3 ? 'text-accent-primary' : 'hover:text-white'}`}
        >
          2. SETUP
        </button> 
        —
        <button 
          onClick={() => leaderboardData && setAppPhase(4)} 
          disabled={!leaderboardData} 
          className={`transition-colors font-semibold ${appPhase === 4 ? 'text-accent-primary' : (leaderboardData ? 'hover:text-white' : 'opacity-50 cursor-not-allowed')}`}
        >
          3. TRAIN & EXPORT
        </button>
      </div>
    </div>
  );

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
        .font-sans { font-family: 'Inter', sans-serif; }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
        
        /* Custom scrollbar for terminal */
        .terminal-scroll::-webkit-scrollbar { width: 6px; }
        .terminal-scroll::-webkit-scrollbar-track { background: transparent; }
        .terminal-scroll::-webkit-scrollbar-thumb { background: #333; border-radius: 10px; }
      `}</style>

      <div className="font-sans antialiased text-text-primary bg-background-primary min-h-screen">
        <Toaster position="bottom-center" toastOptions={{ style: { background: '#1a1a1a', color: '#f5f5f5', border: '1px solid #2a2a2a', fontFamily: 'Inter' } }} />
        
        {/* ==========================================
            VIEW 4: EXPORT SUITE & LEADERBOARD
            ========================================== */}
        {appPhase === 4 && leaderboardData && (
          <div className="flex flex-col items-center pt-16 px-4 pb-24">
            <div className="w-full max-w-5xl">
              <HeaderNav />
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {/* Winner Card */}
                <div className="md:col-span-2 p-8 rounded-xl border-2 border-accent-primary bg-accent-primary/5 relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-8 opacity-10">
                    <Trophy size={160} />
                  </div>
                  <h3 className="text-accent-primary font-bold tracking-wider text-sm mb-2 uppercase flex items-center gap-2">
                    <Trophy size={16}/> Top Performing Model
                  </h3>
                  <h1 className="text-4xl font-bold mb-8 tracking-tight">{leaderboardData[0].name}</h1>
                  
                  <div className="flex gap-8">
                    <div>
                      <p className="text-text-secondary text-sm mb-1 font-medium">Accuracy</p>
                      <p className="text-4xl font-mono font-bold text-accent-success">
                        {(leaderboardData[0].accuracy * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-text-secondary text-sm mb-1 font-medium">F1 Score</p>
                      <p className="text-4xl font-mono font-bold text-text-primary">
                        {(leaderboardData[0].f1_score * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-text-secondary text-sm mb-1 font-medium">Train Time</p>
                      <p className="text-4xl font-mono font-bold text-text-primary">
                        {leaderboardData[0].train_time}s
                      </p>
                    </div>
                  </div>
                </div>

                {/* THE EXPORT BAY */}
                <div className="p-6 rounded-xl border border-border-subtle bg-background-surface flex flex-col justify-between">
                  <div>
                    <h3 className="font-bold text-lg mb-1 tracking-tight flex items-center gap-2">
                      <Download size={18} className="text-accent-primary"/> The Export Suite
                    </h3>
                    <p className="text-sm text-text-secondary mb-6 leading-relaxed">
                      Download your production-ready artifacts to deploy this pipeline anywhere.
                    </p>
                  </div>
                  
                  <div className="space-y-3">
                    <button 
                      onClick={() => window.open(leaderboardData[0].download_url, '_blank')} 
                      className="w-full flex items-center justify-between px-4 py-3 bg-background-elevated border border-border-subtle hover:border-accent-primary hover:bg-accent-primary/10 transition-all rounded-lg group"
                    >
                      <span className="flex items-center gap-2 font-medium text-sm">
                        <Terminal size={16} className="text-indigo-400"/> Model Weights
                      </span>
                      <span className="text-xs font-mono text-text-muted bg-background-primary px-2 py-1 rounded group-hover:text-white">
                        .joblib
                      </span>
                    </button>
                    
                    <button 
                      onClick={() => window.open(notebookUrl, '_blank')} 
                      className="w-full flex items-center justify-between px-4 py-3 bg-background-elevated border border-border-subtle hover:border-accent-primary hover:bg-accent-primary/10 transition-all rounded-lg group"
                    >
                      <span className="flex items-center gap-2 font-medium text-sm">
                        <FileCode2 size={16} className="text-amber-400"/> Python Notebook
                      </span>
                      <span className="text-xs font-mono text-text-muted bg-background-primary px-2 py-1 rounded group-hover:text-white">
                        .ipynb
                      </span>
                    </button>
                    
                    <button 
                      onClick={() => window.open(sessionData.eda_download_url, '_blank')} 
                      className="w-full flex items-center justify-between px-4 py-3 bg-background-elevated border border-border-subtle hover:border-accent-primary hover:bg-accent-primary/10 transition-all rounded-lg group"
                    >
                      <span className="flex items-center gap-2 font-medium text-sm">
                        <FileSpreadsheet size={16} className="text-emerald-400"/> Detailed EDA Report
                      </span>
                      <span className="text-xs font-mono text-text-muted bg-background-primary px-2 py-1 rounded group-hover:text-white">
                        .md
                      </span>
                    </button>
                  </div>
                </div>
              </div>

              <h3 className="text-lg font-bold mb-4 flex items-center gap-2 tracking-tight">
                <BarChart3 size={18} className="text-accent-primary"/> Full Leaderboard
              </h3>
              
              <div className="w-full rounded-xl border border-border-subtle bg-background-surface overflow-hidden">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-border-subtle bg-background-primary">
                      <th className="p-4 font-semibold text-text-secondary">Rank</th>
                      <th className="p-4 font-semibold text-text-secondary">Algorithm</th>
                      <th className="p-4 font-semibold text-text-secondary">Accuracy</th>
                      <th className="p-4 font-semibold text-text-secondary">F1 Score</th>
                      <th className="p-4 font-semibold text-text-secondary flex items-center gap-2"><Timer size={16}/> Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboardData.map((model: any, index: number) => (
                      <tr key={model.id} className="border-b border-border-subtle hover:bg-background-elevated transition-colors">
                        <td className="p-4 font-mono text-text-muted">#{index + 1}</td>
                        <td className="p-4 font-medium tracking-tight">{model.name}</td>
                        <td className="p-4 font-mono font-medium text-white">{(model.accuracy * 100).toFixed(2)}%</td>
                        <td className="p-4 font-mono">{(model.f1_score * 100).toFixed(2)}%</td>
                        <td className="p-4 font-mono text-text-muted">{model.train_time}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ==========================================
            VIEW 3: COMMAND CENTER (Setup)
            ========================================== */}
        {appPhase === 3 && sessionData && (
          <div className="flex flex-col items-center pt-16 px-4 pb-24 relative">
            <div className="w-full max-w-4xl">
              
              {/* DYNAMIC EXECUTION CONSOLE OVERLAY */}
              {isTraining && (
                <div className="fixed inset-0 z-50 bg-background-primary/95 backdrop-blur-md flex flex-col items-center justify-center p-4">
                  <div className="w-full max-w-3xl flex flex-col items-center">
                    <Loader2 size={48} className="animate-spin text-accent-primary mb-6" />
                    <h2 className="text-3xl font-bold mb-2 tracking-tight text-white">Compiling Pipeline</h2>
                    <p className="text-text-secondary mb-12">Please wait while the engine executes your configurations.</p>
                    
                    {/* The Terminal Window */}
                    <div className="w-full bg-[#0D0D12] border border-border-subtle rounded-xl overflow-hidden shadow-2xl">
                      {/* Mac-style header */}
                      <div className="bg-[#1A1A24] px-4 py-3 flex items-center gap-2 border-b border-border-subtle">
                        <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
                        <div className="w-3 h-3 rounded-full bg-amber-500/80"></div>
                        <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
                        <span className="ml-4 text-xs font-mono text-text-muted">kyd-ai-execution-engine ~ bash</span>
                      </div>
                      
                      {/* Terminal Output */}
                      <div className="p-6 h-64 overflow-y-auto terminal-scroll font-mono text-sm leading-relaxed text-indigo-300">
                        {trainingLogs.map((log, i) => (
                          <div key={i} className="mb-2 flex items-start gap-3">
                            <span className="text-text-muted opacity-50 select-none">&gt;</span>
                            <span className={log.includes('[ERROR]') ? 'text-red-400' : log.includes('[SUCCESS]') ? 'text-green-400 font-bold' : ''}>
                              {log}
                            </span>
                          </div>
                        ))}
                        {/* Blinking cursor */}
                        <div className="animate-pulse w-2 h-4 bg-indigo-500 mt-2 ml-6"></div>
                        <div ref={logsEndRef} />
                      </div>

                      {/* Progress Bar */}
                      <div className="h-1.5 w-full bg-[#1A1A24]">
                        <div 
                          className="h-full bg-accent-primary transition-all duration-500 ease-out shadow-[0_0_10px_rgba(99,102,241,0.8)]"
                          style={{ width: `${trainingProgress}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              <HeaderNav />
              
              <div className="space-y-6">
                
                {/* 1. Target Column */}
                <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
                  <h3 className="text-lg font-bold mb-4 flex items-center gap-2 tracking-tight">
                    <Target size={18} className="text-accent-primary"/> 1. Confirm Target Column
                  </h3>
                  <select 
                    className="w-full p-3 rounded-lg bg-background-primary border border-border-subtle text-text-primary focus:border-accent-primary outline-none font-medium" 
                    value={targetColumn} 
                    onChange={(e) => setTargetColumn(e.target.value)}
                  >
                    <option value="">-- Select a column --</option>
                    {sessionData.column_names?.map((col: string) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>

                {/* 2. Drop Columns */}
                <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
                  <h3 className="text-lg font-bold mb-4 flex items-center gap-2 tracking-tight">
                    <Filter size={18} className="text-accent-primary"/> 2. Drop Columns
                  </h3>
                  <p className="text-sm text-text-secondary mb-4">Click to exclude features from the ML pipeline (e.g. unique IDs, unparsed text).</p>
                  <div className="flex flex-wrap gap-2">
                    {sessionData.column_names?.filter((c: string) => c !== targetColumn && c !== timestampColumn).map((col: string) => (
                      <button 
                        key={col} 
                        onClick={() => toggleDropColumn(col)}
                        className={`px-4 py-2 rounded-lg text-sm border transition-all duration-200 flex items-center gap-2 font-medium ${
                          droppedColumns.includes(col) 
                          ? 'bg-red-500/10 border-red-500/50 text-red-400' 
                          : 'bg-background-primary border-border-subtle text-text-muted hover:border-border-active'
                        }`}
                      >
                        {col} {droppedColumns.includes(col) && <SearchX size={14} />}
                      </button>
                    ))}
                  </div>
                </div>

                {/* 3. Engine Mode */}
                <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
                  <h3 className="text-lg font-bold mb-4 flex items-center gap-2 tracking-tight">
                    <Settings2 size={18} className="text-accent-primary"/> 3. Choose Engine Mode
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div 
                      onClick={() => setEngineMode('standard')} 
                      className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                        engineMode === 'standard' ? 'border-accent-primary bg-accent-primary/5' : 'border-border-subtle hover:border-border-active'
                      }`}
                    >
                      <h4 className="font-medium mb-1">Standard Mode</h4>
                      <p className="text-sm text-text-secondary">Random train/test splits. Best for general classification.</p>
                    </div>
                    
                    <div 
                      onClick={() => setEngineMode('timeseries')} 
                      className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                        engineMode === 'timeseries' ? 'border-accent-primary bg-accent-primary/5' : 'border-border-subtle hover:border-border-active'
                      }`}
                    >
                      <h4 className="font-medium mb-1">Time-Series / Trading</h4>
                      <p className="text-sm text-text-secondary">Sequential temporal validation.</p>
                    </div>
                  </div>
                </div>

                {/* 4. Preprocessing & Algorithms */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2 tracking-tight">
                      <Zap size={18} className="text-accent-primary"/> Advanced Preprocessing
                    </h3>
                    <div className="space-y-3">
                      {Object.entries(selectedFeatures).map(([key, value]) => (
                        <label key={key} className="flex items-center gap-3 cursor-pointer group" onClick={() => toggleFeature(key as any)}>
                          <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${
                            value ? 'bg-accent-primary border-accent-primary' : 'border-border-subtle group-hover:border-border-active'
                          }`}>
                            {value && <CheckCircle2 size={14} className="text-white" />}
                          </div>
                          <span className="text-sm text-text-primary capitalize font-medium">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  
                  <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2 tracking-tight">
                      <LayoutList size={18} className="text-accent-primary"/> Algorithms to Train
                    </h3>
                    <div className="space-y-3">
                      {Object.entries(selectedAlgos).map(([key, value]) => (
                        <label key={key} className="flex items-center gap-3 cursor-pointer group" onClick={() => toggleAlgo(key as any)}>
                          <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${
                            value ? 'bg-accent-primary border-accent-primary' : 'border-border-subtle group-hover:border-border-active'
                          }`}>
                            {value && <CheckCircle2 size={14} className="text-white" />}
                          </div>
                          <span className="text-sm text-text-primary capitalize font-medium">{key.replace('_', ' ')}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="flex justify-end items-center mt-8 pt-6 border-t border-border-subtle">
                  <button 
                    onClick={handleRunAutoML} 
                    className="flex items-center gap-2 px-8 py-4 bg-accent-primary hover:bg-indigo-600 text-white rounded-lg font-bold tracking-wide transition-all shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)]"
                  >
                    <Play size={18} fill="currentColor" /> Run Preprocessing & Train
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==========================================
            VIEW 2: EDA HUB
            ========================================== */}
        {appPhase === 2 && sessionData && (
          <div className="flex flex-col items-center pt-16 px-4 pb-24">
            <div className="w-full max-w-6xl">
              <HeaderNav />

              {/* AI Insights */}
              {sessionData.eda.smart_insights && sessionData.eda.smart_insights.length > 0 && (
                <div className="mb-8 p-5 rounded-xl border border-accent-primary/30 bg-accent-primary/5 flex flex-col gap-4 shadow-inner">
                  <h3 className="text-accent-primary font-bold flex items-center gap-2 text-sm uppercase tracking-wider">
                    <Lightbulb size={16}/> AI Feature Engineering Insights
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {sessionData.eda.smart_insights.map((insight: any, idx: number) => (
                      <div key={idx} className={`p-4 rounded-lg border flex items-start gap-3 bg-background-surface ${
                        insight.type === 'danger' ? 'border-red-500/50 text-red-100' : 
                        insight.type === 'warning' ? 'border-amber-500/50 text-amber-100' : 
                        'border-indigo-500/50 text-indigo-100'
                      }`}>
                        {insight.type === 'danger' ? <AlertTriangle size={18} className="text-red-400 mt-0.5 shrink-0"/> : 
                         insight.type === 'warning' ? <AlertTriangle size={18} className="text-amber-400 mt-0.5 shrink-0"/> : 
                         <CheckCircle2 size={18} className="text-indigo-400 mt-0.5 shrink-0"/>}
                        <p className="text-sm leading-relaxed font-medium">{insight.message}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Banner Overview */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                <div className="p-6 rounded-xl border border-border-subtle bg-background-surface flex flex-col gap-1">
                  <span className="text-text-muted text-sm font-semibold flex items-center gap-2 uppercase tracking-wider">
                    <Database size={14}/> Dataset Size
                  </span>
                  <span className="text-3xl font-bold font-mono">
                    {sessionData.rows.toLocaleString()} <span className="text-sm font-sans text-text-secondary font-normal">rows</span>
                  </span>
                  <span className="text-text-muted text-xs mt-1 font-mono">
                    {sessionData.columns} cols • {sessionData.memory_mb.toFixed(2)} MB
                  </span>
                </div>
                
                <div className="p-6 rounded-xl border border-border-subtle bg-background-surface flex flex-col gap-1">
                  <span className="text-text-muted text-sm font-semibold flex items-center gap-2 uppercase tracking-wider">
                    <Activity size={14}/> Data Issues
                  </span>
                  <span className="text-3xl font-bold font-mono text-accent-warning">
                    {sessionData.eda.missing_summary.length} <span className="text-sm font-sans text-text-secondary font-normal">missing cols</span>
                  </span>
                  <span className="text-text-muted text-xs mt-1 font-mono">
                    {sessionData.eda.duplicate_count} duplicate rows
                  </span>
                </div>
                
                <div className="p-6 rounded-xl border border-accent-success/30 bg-accent-success/5 flex flex-col gap-1 relative overflow-hidden md:col-span-2">
                  <span className="text-text-muted text-sm font-semibold flex items-center gap-2 z-10 uppercase tracking-wider">
                    <Activity size={14}/> Health Score
                  </span>
                  <div className="flex items-baseline gap-2 z-10">
                    <span className={`text-5xl font-bold font-mono tracking-tighter ${
                      sessionData.health_score > 70 ? 'text-accent-success' : 
                      sessionData.health_score > 40 ? 'text-accent-warning' : 
                      'text-red-500'
                    }`}>
                      {sessionData.health_score}
                    </span>
                    <span className="text-text-secondary font-mono">/100</span>
                  </div>
                  <div className={`absolute bottom-0 left-0 h-1.5 transition-all duration-1000 ${
                    sessionData.health_score > 70 ? 'bg-accent-success' : 
                    sessionData.health_score > 40 ? 'bg-accent-warning' : 
                    'bg-red-500'
                  }`} style={{ width: `${sessionData.health_score}%` }}></div>
                </div>
              </div>

              {/* Bivariate Target Selector */}
              <div className="mb-8 p-6 rounded-xl border border-indigo-500/30 bg-indigo-500/5 flex flex-col md:flex-row items-center justify-between gap-4">
                <div>
                  <h3 className="text-lg font-bold flex items-center gap-2 text-indigo-400 tracking-tight">
                    <SplitSquareVertical size={18}/> Target Variable Analysis
                  </h3>
                  <p className="text-sm text-text-secondary mt-1">Select your predictive target to unlock stacked Bivariate analysis.</p>
                </div>
                <div className="flex items-center gap-3 w-full md:w-auto">
                  {isFetchingBivariate && <Loader2 size={18} className="animate-spin text-accent-primary" />}
                  <select 
                    className="w-full md:w-64 p-3 rounded-lg bg-background-surface border border-indigo-500/30 text-white focus:border-accent-primary outline-none font-medium" 
                    value={edaTarget} 
                    onChange={(e) => handleEdaTargetSelect(e.target.value)}
                  >
                    <option value="">-- No Target Selected --</option>
                    {sessionData.column_names?.map((col: string) => (
                      <option key={`target-${col}`} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Missing Data Action Center */}
              {sessionData.eda.missing_summary.length > 0 && (
                <div className="mb-8 p-6 rounded-xl border border-accent-warning/30 bg-background-surface">
                  <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-accent-warning tracking-tight">
                    <SearchX size={18}/> Missing Data Action Center
                  </h3>
                  <div className="w-full rounded-lg border border-border-subtle overflow-hidden">
                    <table className="w-full text-left border-collapse bg-background-primary">
                      <thead>
                        <tr className="border-b border-border-subtle">
                          <th className="p-4 font-semibold text-text-secondary text-sm">Column Name</th>
                          <th className="p-4 font-semibold text-text-secondary text-sm">Missing %</th>
                          <th className="p-4 font-semibold text-text-secondary text-sm">Data Type</th>
                          <th className="p-4 font-semibold text-text-secondary text-sm">Imputation Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sessionData.eda.missing_summary.map((col: any) => (
                          <tr key={col.column} className="border-b border-border-subtle last:border-0 hover:bg-background-elevated transition-colors">
                            <td className="p-4 font-mono font-medium">{col.column}</td>
                            <td className="p-4 font-mono text-accent-warning font-medium">{col.missing_pct}%</td>
                            <td className="p-4 text-sm text-text-muted font-mono">{col.dtype}</td>
                            <td className="p-4">
                              <select 
                                className="w-full p-2 rounded-md bg-background-surface border border-border-subtle text-sm font-medium focus:border-accent-primary outline-none" 
                                value={imputationStrategy[col.column] || ''} 
                                onChange={(e) => handleImputationChange(col.column, e.target.value)}
                              >
                                {col.dtype.includes('float') || col.dtype.includes('int') ? (
                                  <>
                                    <option value="median">Fill with Median</option>
                                    <option value="mean">Fill with Mean</option>
                                    <option value="knn">KNN Imputation (Neighbors: 5)</option>
                                  </>
                                ) : (
                                  <>
                                    <option value="mode">Fill with Mode</option>
                                    <option value="constant">Fill with 'Unknown'</option>
                                  </>
                                )}
                                <option value="drop_rows">Drop Rows</option>
                                <option value="drop_col">Drop Column</option>
                              </select>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Distributions Grid */}
              <h3 className="text-lg font-bold mb-4 flex items-center gap-2 tracking-tight">
                <FileSpreadsheet size={18} className="text-accent-primary"/> Column Distributions
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                {sessionData.eda.column_stats?.map((col: any) => {
                  const isTargetCol = edaTarget === col.name;
                  const hasBivariate = bivariateData && bivariateData.data[col.name];
                  const chartData = hasBivariate ? bivariateData.data[col.name] : col.chart_data;

                  return (
                    <div 
                      key={col.name} 
                      className={`p-5 rounded-xl border transition-all duration-300 shadow-sm group ${
                        isTargetCol ? 'border-indigo-500 bg-indigo-500/5' : 'border-border-subtle bg-background-surface hover:-translate-y-1 hover:border-border-active hover:shadow-lg'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-4">
                        <div>
                          <h4 className="font-mono font-bold text-text-primary truncate max-w-[150px] flex items-center gap-2" title={col.name}>
                            {col.name} 
                            {isTargetCol && <span className="text-[10px] uppercase bg-indigo-500 font-sans text-white px-2 py-0.5 rounded tracking-wider">Target</span>}
                          </h4>
                          <span className="text-xs text-text-muted px-2 py-0.5 bg-background-primary rounded border border-border-subtle font-mono">{col.dtype}</span>
                        </div>
                        {col.missing > 0 && <span className="text-xs text-accent-warning font-medium flex items-center gap-1"><AlertTriangle size={12}/> {col.missing} nulls</span>}
                      </div>

                      <div className="h-32 w-full mb-4 opacity-80 group-hover:opacity-100 transition-opacity">
                        <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                          <BarChart data={chartData}>
                            <RechartsTooltip 
                              contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #3a3a3a', borderRadius: '8px', fontSize: '12px', fontFamily: 'Inter' }} 
                              cursor={{ fill: '#2a2a2a' }} 
                            />
                            {hasBivariate ? (
                              bivariateData.target_classes.map((tClass: string, idx: number) => (
                                <Bar 
                                  key={tClass} 
                                  dataKey={tClass} 
                                  stackId="a" 
                                  fill={CHART_COLORS[idx % CHART_COLORS.length]} 
                                  radius={idx === bivariateData.target_classes.length - 1 ? [2, 2, 0, 0] : [0,0,0,0]} 
                                />
                              ))
                            ) : (
                              <Bar 
                                dataKey="count" 
                                fill={isTargetCol ? "#3b82f6" : "#6366f1"} 
                                radius={[2, 2, 0, 0]} 
                              />
                            )}
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs text-text-secondary border-t border-border-subtle pt-3 font-mono">
                        {col.type === 'numeric' ? (
                          <>
                            <div><span className="text-text-muted">Mean:</span> {col.mean ?? '---'}</div>
                            <div><span className="text-text-muted">Median:</span> {col.median ?? '---'}</div>
                            <div><span className="text-text-muted">Min:</span> {col.min ?? '---'}</div>
                            <div><span className="text-text-muted">Max:</span> {col.max ?? '---'}</div>
                          </>
                        ) : (
                          <div className="col-span-2"><span className="text-text-muted">Unique Categories:</span> {col.unique}</div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Correlation Matrix */}
              {sessionData.eda.correlation_matrix?.columns && sessionData.eda.correlation_matrix.columns.length > 0 && (
                <div className="mb-8">
                  <h3 className="text-lg font-bold mb-4 flex items-center gap-2 tracking-tight">
                    <Network size={18} className="text-accent-primary"/> Feature Correlation Matrix
                  </h3>
                  <div className="p-6 rounded-xl border border-border-subtle bg-background-surface overflow-x-auto">
                    <div className="flex items-start justify-between mb-4">
                      <p className="text-sm text-text-secondary">
                        Visual map of linear relationships. <span className="text-indigo-400 font-medium">Blue = Positive</span>, <span className="text-red-400 font-medium">Red = Negative</span>.
                      </p>
                      <p className="text-xs text-accent-warning flex items-center gap-1"><Info size={14}/> Highlighted cells (|r| &gt; 0.85) indicate potential multi-collinearity.</p>
                    </div>
                    
                    <div className="inline-block min-w-full">
                      <table className="border-collapse text-xs font-mono">
                        <thead>
                          <tr>
                            <th className="p-2 border border-border-subtle"></th>
                            {sessionData.eda.correlation_matrix.columns.map((col: string) => (
                              <th key={`head-${col}`} className="p-2 border border-border-subtle text-text-muted font-normal max-w-[100px] truncate" title={col}>
                                {col.substring(0, 8)}..
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {sessionData.eda.correlation_matrix.values.map((row: number[], i: number) => (
                            <tr key={`row-${i}`}>
                              <th className="p-2 border border-border-subtle text-text-muted font-normal text-left max-w-[100px] truncate" title={sessionData.eda.correlation_matrix.columns[i]}>
                                {sessionData.eda.correlation_matrix.columns[i].substring(0, 10)}..
                              </th>
                              {row.map((val: number, j: number) => {
                                const isHighlyCorrelated = Math.abs(val) > 0.85 && i !== j;
                                return (
                                  <td 
                                    key={`cell-${i}-${j}`} 
                                    title={`${sessionData.eda.correlation_matrix.columns[i]} ↔ ${sessionData.eda.correlation_matrix.columns[j]}: r = ${val.toFixed(2)}`}
                                    className={`w-12 h-12 p-0 text-center relative group cursor-crosshair
                                      ${isHighlyCorrelated ? 'border-2 border-accent-warning z-10 shadow-[0_0_10px_rgba(245,158,11,0.5)]' : 'border border-border-subtle'}
                                    `}
                                  >
                                    <div className={`absolute inset-0 ${getCorrelationColor(val)}`} style={{ opacity: getCorrelationOpacity(val) }}></div>
                                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-background-elevated border border-accent-primary z-20 transition-opacity rounded">
                                      {val.toFixed(2)}
                                    </div>
                                  </td>
                                )
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              <div className="flex justify-end items-center mt-8 pt-6 border-t border-border-subtle">
                <button 
                  onClick={() => setAppPhase(3)} 
                  className="flex items-center gap-2 px-8 py-4 bg-accent-primary hover:bg-indigo-600 text-white rounded-lg font-bold tracking-wide transition-all shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)]"
                >
                  Proceed to ML Setup <ChevronRight size={20} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ==========================================
            VIEW 1: UPLOAD ZONE
            ========================================== */}
        {appPhase === 1 && (
          <div className="flex flex-col items-center pt-32 px-4 relative">
            <div className="text-center mb-12">
              <h1 className="text-5xl font-bold tracking-tighter mb-4 text-text-primary">
                KyD<span className="text-accent-primary">.ai</span>
              </h1>
              <p className="text-text-secondary text-lg font-medium">
                Upload your dataset to initialize the engine.
              </p>
            </div>

            <div 
              className={`w-full max-w-2xl p-16 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center transition-all duration-300 ease-in-out bg-background-surface
                ${isDragging && !isUploading ? 'border-accent-primary bg-background-elevated scale-[1.02]' : ''}
                ${!isUploading ? 'border-border-subtle hover:border-border-active hover:bg-background-elevated cursor-pointer' : 'border-border-subtle opacity-80 cursor-not-allowed'}
              `} 
              onDragOver={handleDragOver} 
              onDragLeave={handleDragLeave} 
              onDrop={handleDrop} 
              onClick={() => !isUploading && fileInputRef.current?.click()}
            >
              <input type="file" accept=".csv" className="hidden" ref={fileInputRef} onChange={handleFileInput} />
              
              <div className={`p-5 rounded-full mb-6 transition-colors ${isDragging && !isUploading ? 'bg-accent-primary/20 text-accent-primary' : 'bg-background-primary text-text-muted'}`}>
                {isUploading ? <Loader2 size={48} className="animate-spin text-accent-primary" /> : <UploadCloud size={48} />}
              </div>
              
              <h3 className="text-2xl font-bold tracking-tight mb-2">
                {isUploading ? 'Crunching the numbers...' : isDragging ? 'Drop it right here' : 'Drag & Drop your CSV file'}
              </h3>
              <p className="text-text-muted font-medium mb-8">
                {isUploading ? 'Building data intelligence profiles' : 'or click to browse local files'}
              </p>
              
              {!isUploading && (
                <div className="flex items-center gap-2 text-sm text-text-secondary font-mono bg-background-primary px-4 py-2 rounded-lg border border-border-subtle">
                  <FileSpreadsheet size={16} /><span>Supports: .csv • Max size: 500MB</span>
                </div>
              )}

              {isUploading && (
                <div className="w-full max-w-md mt-6">
                  <div className="h-2 w-full bg-background-primary rounded-full overflow-hidden border border-border-subtle">
                    <div className="h-full bg-accent-primary transition-all duration-300 ease-out" style={{ width: `${uploadProgress}%` }} />
                  </div>
                  <p className="text-center text-sm font-mono text-text-secondary mt-3">{uploadProgress}%</p>
                </div>
              )}
            </div>
            
            {/* Resume Session Button */}
            {sessionData && !isUploading && (
              <button 
                onClick={() => setAppPhase(2)} 
                className="mt-8 px-6 py-3 rounded-lg border border-accent-primary text-accent-primary hover:bg-accent-primary/10 font-bold transition-all flex items-center gap-2"
              >
                Resume Active Session <ChevronRight size={18} />
              </button>
            )}
          </div>
        )}
      </div>
    </>
  );
}