import { useState, useRef } from 'react';
import {
  UploadCloud, FileSpreadsheet, Clock, Loader2, Database,
  Activity, CheckCircle2, ChevronRight, Settings2, Zap,
  LayoutList, Target, Play, Trophy, BarChart3, Timer,
  Download, AlertTriangle, Network, SearchX, Info, Lightbulb, SplitSquareVertical
} from 'lucide-react';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import { BarChart, Bar, ResponsiveContainer, Tooltip as RechartsTooltip, XAxis, Legend } from 'recharts';

export default function App() {
  const [appPhase, setAppPhase] = useState(1);
  const [sessionData, setSessionData] = useState<any>(null);
  const [leaderboardData, setLeaderboardData] = useState<any>(null);

  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [targetColumn, setTargetColumn] = useState('');
  const [engineMode, setEngineMode] = useState<'standard' | 'timeseries'>('standard');
  const [timestampColumn, setTimestampColumn] = useState('');
  const [imputationStrategy, setImputationStrategy] = useState<Record<string, string>>({});

  // NEW: Bivariate Analysis State
  const [edaTarget, setEdaTarget] = useState('');
  const [bivariateData, setBivariateData] = useState<any>(null);
  const [isFetchingBivariate, setIsFetchingBivariate] = useState(false);

  const [selectedFeatures, setSelectedFeatures] = useState({ extractDates: true, scaleNumeric: true, removeLowVariance: true });
  const [selectedAlgos, setSelectedAlgos] = useState({ xgboost: true, lightgbm: true, random_forest: true, svm: false, logistic: true });

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
    await performUpload(file);
  };

  const performUpload = async (file: File) => {
    setIsUploading(true); setUploadProgress(0);
    const formData = new FormData(); formData.append('file', file);
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
      setIsUploading(false); setUploadProgress(0);
    }
  };

  // NEW: Fetch Bivariate Data when Target is selected in EDA Phase
  const handleEdaTargetSelect = async (selectedTarget: string) => {
    setEdaTarget(selectedTarget);
    setTargetColumn(selectedTarget); // Sync it with Phase 3 automatically

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

  const handleRunAutoML = async () => {
    if (!targetColumn) return toast.error("Please select a target column to predict.");
    if (engineMode === 'timeseries' && !timestampColumn) return toast.error("Time-Series mode requires a timestamp column.");

    setIsTraining(true);
    toast.success("AutoML Engine Initialized. Training models...");

    try {
      const response = await axios.post('http://localhost:8000/api/train', {
        dataset_id: sessionData.dataset_id,
        target_column: targetColumn,
        engine_mode: engineMode,
        features: selectedFeatures,
        algos: selectedAlgos,
      });
      setLeaderboardData(response.data.leaderboard);
      setAppPhase(4);
      toast.success("Training complete!");
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Training failed');
    } finally {
      setIsTraining(false);
    }
  };

  const getCorrelationColor = (val: number) => {
    if (val === 1) return 'bg-background-elevated text-text-muted';
    if (val > 0) return `bg-indigo-500 text-white font-medium`;
    if (val < 0) return `bg-red-500 text-white font-medium`;
    return 'bg-background-surface text-text-primary';
  };

  const getCorrelationOpacity = (val: number) => Math.max(0.15, Math.abs(val));

  // Palette for stacked charts
  const CHART_COLORS = ['#ef4444', '#22c55e', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899'];

  // ==========================================
  // VIEW 4: LEADERBOARD & EXPORT
  // ==========================================
  if (appPhase === 4 && leaderboardData) {
    const bestModel = leaderboardData[0];
    return (
      <div className="min-h-screen flex flex-col items-center pt-16 px-4 pb-24">
        <Toaster position="bottom-center" toastOptions={{ style: { background: '#1a1a1a', color: '#f5f5f5', border: '1px solid #2a2a2a' } }} />
        <div className="w-full max-w-5xl">
          <div className="flex items-center justify-between mb-8 pb-4 border-b border-border-subtle">
            <h2 className="text-2xl font-bold flex items-center gap-2">KyD<span className="text-accent-primary">.ai</span><span className="text-text-muted font-normal text-xl mx-2">/</span><span className="text-text-secondary font-medium text-xl">Results</span></h2>
            <div className="flex items-center gap-2 text-sm text-text-muted font-mono tracking-wider">1. EDA — 2. SETUP — <span className="text-accent-primary">3. TRAIN</span> — 4. EXPORT</div>
          </div>
          <div className="p-8 rounded-xl border-2 border-accent-primary bg-accent-primary/5 mb-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-8 opacity-10"><Trophy size={120} /></div>
            <h3 className="text-accent-primary font-bold tracking-wider text-sm mb-2 uppercase flex items-center gap-2"><Trophy size={16} /> Top Performing Model</h3>
            <h1 className="text-4xl font-bold mb-6">{bestModel.name}</h1>
            <div className="flex gap-8 mb-8">
              <div><p className="text-text-secondary text-sm mb-1">Accuracy</p><p className="text-3xl font-mono text-accent-success">{(bestModel.accuracy * 100).toFixed(2)}%</p></div>
              <div><p className="text-text-secondary text-sm mb-1">F1 Score</p><p className="text-3xl font-mono text-text-primary">{(bestModel.f1_score * 100).toFixed(2)}%</p></div>
              <div><p className="text-text-secondary text-sm mb-1">Train Time</p><p className="text-3xl font-mono text-text-primary">{bestModel.train_time}s</p></div>
            </div>
            <button onClick={() => window.open(bestModel.download_url, '_blank')} className="relative z-10 flex items-center gap-2 px-6 py-3 bg-background-primary border border-border-subtle hover:border-accent-primary hover:bg-accent-primary/10 text-white rounded-lg font-medium transition-all duration-300 shadow-md hover:shadow-[0_0_15px_rgba(99,102,241,0.4)] group">
              <Download size={18} className="group-hover:-translate-y-1 transition-transform" /> Download .joblib Model
            </button>
          </div>
          <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><BarChart3 size={18} className="text-accent-primary" /> Full Leaderboard</h3>
          <div className="w-full rounded-xl border border-border-subtle bg-background-surface overflow-hidden">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-border-subtle bg-background-primary">
                  <th className="p-4 font-medium text-text-secondary">Rank</th>
                  <th className="p-4 font-medium text-text-secondary">Algorithm</th>
                  <th className="p-4 font-medium text-text-secondary">Accuracy</th>
                  <th className="p-4 font-medium text-text-secondary">F1 Score</th>
                  <th className="p-4 font-medium text-text-secondary flex items-center gap-2"><Timer size={16} /> Time</th>
                </tr>
              </thead>
              <tbody>
                {leaderboardData.map((model: any, index: number) => (
                  <tr key={model.id} className="border-b border-border-subtle hover:bg-background-elevated transition-colors">
                    <td className="p-4 font-mono text-text-muted">#{index + 1}</td>
                    <td className="p-4 font-medium">{model.name}</td>
                    <td className="p-4 font-mono">{(model.accuracy * 100).toFixed(2)}%</td>
                    <td className="p-4 font-mono">{(model.f1_score * 100).toFixed(2)}%</td>
                    <td className="p-4 font-mono text-text-muted">{model.train_time}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }

  // ==========================================
  // VIEW 3: COMMAND CENTER (Setup)
  // ==========================================
  if (appPhase === 3 && sessionData) {
    return (
      <div className="min-h-screen flex flex-col items-center pt-16 px-4 pb-24">
        <Toaster position="bottom-center" toastOptions={{ style: { background: '#1a1a1a', color: '#f5f5f5', border: '1px solid #2a2a2a' } }} />
        <div className="w-full max-w-4xl relative">

          {isTraining && (
            <div className="absolute inset-0 z-50 bg-background-primary/80 backdrop-blur-sm flex flex-col items-center justify-center rounded-xl border border-border-subtle">
              <Loader2 size={48} className="animate-spin text-accent-primary mb-4" />
              <h2 className="text-xl font-bold mb-2">Training Models...</h2>
              <p className="text-text-secondary">Running Machine Learning algorithms on your data.</p>
            </div>
          )}

          <div className="flex items-center justify-between mb-8 pb-4 border-b border-border-subtle">
            <h2 className="text-2xl font-bold flex items-center gap-2">KyD<span className="text-accent-primary">.ai</span><span className="text-text-muted font-normal text-xl mx-2">/</span><span className="text-text-secondary font-medium text-xl">Setup</span></h2>
            <div className="flex items-center gap-2 text-sm text-text-muted font-mono tracking-wider">1. EDA — <span className="text-accent-primary">2. SETUP</span> — 3. TRAIN — 4. EXPORT</div>
          </div>

          <div className="space-y-6">
            <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
              <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><Target size={18} className="text-accent-primary" /> 1. Confirm Target Column</h3>
              <select className="w-full p-3 rounded-lg bg-background-primary border border-border-subtle text-text-primary focus:border-accent-primary outline-none" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
                <option value="">-- Select a column --</option>
                {sessionData.column_names?.map((col: string) => <option key={col} value={col}>{col}</option>)}
              </select>
            </div>

            <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
              <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><Settings2 size={18} className="text-accent-primary" /> 2. Choose Engine Mode</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div onClick={() => setEngineMode('standard')} className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${engineMode === 'standard' ? 'border-accent-primary bg-accent-primary/5' : 'border-border-subtle hover:border-border-active'}`}>
                  <h4 className="font-medium mb-1">Standard Mode</h4>
                  <p className="text-sm text-text-secondary">Random train/test splits. Best for general classification.</p>
                </div>
                <div onClick={() => setEngineMode('timeseries')} className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${engineMode === 'timeseries' ? 'border-accent-primary bg-accent-primary/5' : 'border-border-subtle hover:border-border-active'}`}>
                  <h4 className="font-medium mb-1">Time-Series / Trading</h4>
                  <p className="text-sm text-text-secondary">Sequential temporal validation.</p>
                </div>
              </div>
              {engineMode === 'timeseries' && (
                <div className="mt-4 p-4 rounded-lg bg-background-primary border border-border-subtle">
                  <label className="block text-sm font-medium text-text-secondary mb-2">Select Timestamp Column</label>
                  <select className="w-full p-3 rounded-lg bg-background-surface border border-border-subtle text-text-primary focus:border-accent-primary outline-none" value={timestampColumn} onChange={(e) => setTimestampColumn(e.target.value)}>
                    <option value="">-- Select timestamp column --</option>
                    {sessionData.column_names?.map((col: string) => <option key={col} value={col}>{col}</option>)}
                  </select>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
                <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><Zap size={18} className="text-accent-primary" /> Preprocessing</h3>
                <div className="space-y-3">
                  {Object.entries(selectedFeatures).map(([key, value]) => (
                    <label key={key} className="flex items-center gap-3 cursor-pointer group" onClick={() => toggleFeature(key as any)}>
                      <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${value ? 'bg-accent-primary border-accent-primary' : 'border-border-subtle group-hover:border-border-active'}`}>
                        {value && <CheckCircle2 size={14} className="text-white" />}
                      </div>
                      <span className="text-sm text-text-primary capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                    </label>
                  ))}
                </div>
              </div>
              <div className="p-6 rounded-xl border border-border-subtle bg-background-surface">
                <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><LayoutList size={18} className="text-accent-primary" /> Algorithms</h3>
                <div className="space-y-3">
                  {Object.entries(selectedAlgos).map(([key, value]) => (
                    <label key={key} className="flex items-center gap-3 cursor-pointer group" onClick={() => toggleAlgo(key as any)}>
                      <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${value ? 'bg-accent-primary border-accent-primary' : 'border-border-subtle group-hover:border-border-active'}`}>
                        {value && <CheckCircle2 size={14} className="text-white" />}
                      </div>
                      <span className="text-sm text-text-primary capitalize">{key.replace('_', ' ')}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex justify-between items-center mt-8 pt-6 border-t border-border-subtle">
              <button onClick={() => setAppPhase(2)} className="px-6 py-3 text-text-secondary hover:text-text-primary transition-colors">Back to EDA</button>
              <button onClick={handleRunAutoML} className="flex items-center gap-2 px-8 py-4 bg-accent-primary hover:bg-indigo-600 text-white rounded-lg font-medium transition-all shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)]">
                <Play size={18} fill="currentColor" /> Run AutoML Engine
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ==========================================
  // VIEW 2: EDA HUB (Deep Intelligence)
  // ==========================================
  if (appPhase === 2 && sessionData) {
    return (
      <div className="min-h-screen flex flex-col items-center pt-16 px-4 pb-24">
        <Toaster position="bottom-center" toastOptions={{ style: { background: '#1a1a1a', color: '#f5f5f5', border: '1px solid #2a2a2a' } }} />
        <div className="w-full max-w-6xl">

          <div className="flex items-center justify-between mb-8 pb-4 border-b border-border-subtle">
            <h2 className="text-2xl font-bold flex items-center gap-2">KyD<span className="text-accent-primary">.ai</span><span className="text-text-muted font-normal text-xl mx-2">/</span><span className="text-text-secondary font-medium text-xl">{sessionData.filename}</span></h2>
            <div className="flex items-center gap-2 text-sm text-text-muted font-mono tracking-wider"><span className="text-accent-primary">1. EDA</span> — 2. SETUP — 3. TRAIN — 4. EXPORT</div>
          </div>

          {/* UPGRADE 3: AI Smart Insights Banner */}
          {sessionData.eda.smart_insights && sessionData.eda.smart_insights.length > 0 && (
            <div className="mb-8 p-4 rounded-xl border border-accent-primary/30 bg-accent-primary/5 flex flex-col gap-3 shadow-inner">
              <h3 className="text-accent-primary font-semibold flex items-center gap-2 text-sm uppercase tracking-wider"><Lightbulb size={16} /> AI Feature Engineering Insights</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {sessionData.eda.smart_insights.map((insight: any, idx: number) => (
                  <div key={idx} className={`p-3 rounded-lg border flex items-start gap-3 bg-background-surface ${insight.type === 'danger' ? 'border-red-500/50 text-red-100' : insight.type === 'warning' ? 'border-amber-500/50 text-amber-100' : 'border-indigo-500/50 text-indigo-100'}`}>
                    {insight.type === 'danger' ? <AlertTriangle size={18} className="text-red-400 mt-0.5 shrink-0" /> : insight.type === 'warning' ? <AlertTriangle size={18} className="text-amber-400 mt-0.5 shrink-0" /> : <CheckCircle2 size={18} className="text-indigo-400 mt-0.5 shrink-0" />}
                    <p className="text-sm leading-relaxed">{insight.message}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Banner */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="p-6 rounded-xl border border-border-subtle bg-background-surface flex flex-col gap-1">
              <span className="text-text-muted text-sm font-medium flex items-center gap-2"><Database size={16} /> Dataset Size</span>
              <span className="text-2xl font-bold font-mono">{sessionData.rows.toLocaleString()} <span className="text-sm font-sans text-text-secondary font-normal">rows</span></span>
              <span className="text-text-muted text-xs mt-1">{sessionData.columns} cols • {sessionData.memory_mb.toFixed(2)} MB</span>
            </div>
            <div className="p-6 rounded-xl border border-border-subtle bg-background-surface flex flex-col gap-1">
              <span className="text-text-muted text-sm font-medium flex items-center gap-2"><Activity size={16} /> Data Issues</span>
              <span className="text-2xl font-bold font-mono text-accent-warning">{sessionData.eda.missing_summary.length} <span className="text-sm font-sans text-text-secondary font-normal">missing cols</span></span>
              <span className="text-text-muted text-xs mt-1">{sessionData.eda.duplicate_count} duplicate rows</span>
            </div>
            <div className="p-6 rounded-xl border border-accent-success/30 bg-accent-success/5 flex flex-col gap-1 relative overflow-hidden md:col-span-2">
              <span className="text-text-muted text-sm font-medium flex items-center gap-2 z-10"><Activity size={16} /> Health Score</span>
              <div className="flex items-baseline gap-2 z-10">
                <span className={`text-4xl font-bold font-mono ${sessionData.health_score > 70 ? 'text-accent-success' : sessionData.health_score > 40 ? 'text-accent-warning' : 'text-red-500'}`}>{sessionData.health_score}</span>
                <span className="text-text-secondary">/100</span>
              </div>
              <div className={`absolute bottom-0 left-0 h-1.5 transition-all duration-1000 ${sessionData.health_score > 70 ? 'bg-accent-success' : sessionData.health_score > 40 ? 'bg-accent-warning' : 'bg-red-500'}`} style={{ width: `${sessionData.health_score}%` }}></div>
            </div>
          </div>

          {/* UPGRADE 2: Bivariate Target Selector */}
          <div className="mb-8 p-6 rounded-xl border border-indigo-500/30 bg-indigo-500/5 flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              <h3 className="text-lg font-medium flex items-center gap-2 text-indigo-400"><SplitSquareVertical size={18} /> Target Variable Analysis</h3>
              <p className="text-sm text-text-secondary mt-1">Select your predictive target to unlock stacked Bivariate analysis across all features.</p>
            </div>
            <div className="flex items-center gap-3 w-full md:w-auto">
              {isFetchingBivariate && <Loader2 size={18} className="animate-spin text-accent-primary" />}
              <select
                className="w-full md:w-64 p-3 rounded-lg bg-background-surface border border-indigo-500/30 text-text-primary focus:border-accent-primary outline-none"
                value={edaTarget}
                onChange={(e) => handleEdaTargetSelect(e.target.value)}
              >
                <option value="">-- No Target Selected --</option>
                {sessionData.column_names?.map((col: string) => <option key={`target-${col}`} value={col}>{col}</option>)}
              </select>
            </div>
          </div>

          {/* Missing Data Action Center */}
          {sessionData.eda.missing_summary.length > 0 && (
            <div className="mb-8 p-6 rounded-xl border border-accent-warning/30 bg-background-surface">
              <h3 className="text-lg font-medium mb-4 flex items-center gap-2 text-accent-warning"><SearchX size={18} /> Missing Data Action Center</h3>

              <div className="w-full rounded-lg border border-border-subtle overflow-hidden">
                <table className="w-full text-left border-collapse bg-background-primary">
                  <thead>
                    <tr className="border-b border-border-subtle">
                      <th className="p-4 font-medium text-text-secondary text-sm">Column Name</th>
                      <th className="p-4 font-medium text-text-secondary text-sm">Missing %</th>
                      <th className="p-4 font-medium text-text-secondary text-sm">Data Type</th>
                      <th className="p-4 font-medium text-text-secondary text-sm">Imputation Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sessionData.eda.missing_summary.map((col: any) => (
                      <tr key={col.column} className="border-b border-border-subtle last:border-0 hover:bg-background-elevated transition-colors">
                        <td className="p-4 font-mono font-medium">{col.column}</td>
                        <td className="p-4 font-mono text-accent-warning">{col.missing_pct}%</td>
                        <td className="p-4 text-sm text-text-muted">{col.dtype}</td>
                        <td className="p-4">
                          <select
                            className="w-full p-2 rounded-md bg-background-surface border border-border-subtle text-sm focus:border-accent-primary outline-none"
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

          {/* Distributions Grid (Dynamic Bivariate Support) */}
          <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><FileSpreadsheet size={18} className="text-accent-primary" /> Column Distributions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {sessionData.eda.column_stats?.map((col: any) => {

              // Dynamic Chart selection based on Bivariate API state
              const isTargetCol = edaTarget === col.name;
              const hasBivariate = bivariateData && bivariateData.data[col.name];
              const chartData = hasBivariate ? bivariateData.data[col.name] : col.chart_data;

              return (
                <div key={col.name} className={`p-5 rounded-xl border transition-all duration-300 shadow-sm group ${isTargetCol ? 'border-indigo-500 bg-indigo-500/5' : 'border-border-subtle bg-background-surface hover:-translate-y-1 hover:border-border-active hover:shadow-lg'}`}>
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h4 className="font-mono font-medium text-text-primary truncate max-w-[150px] flex items-center gap-2" title={col.name}>
                        {col.name} {isTargetCol && <span className="text-[10px] uppercase bg-indigo-500 text-white px-1.5 py-0.5 rounded">Target</span>}
                      </h4>
                      <span className="text-xs text-text-muted px-2 py-0.5 bg-background-primary rounded border border-border-subtle">{col.dtype}</span>
                    </div>
                    {col.missing > 0 && <span className="text-xs text-accent-warning flex items-center gap-1"><AlertTriangle size={12} /> {col.missing} nulls</span>}
                  </div>

                  <div className="h-32 w-full mb-4 opacity-80 group-hover:opacity-100 transition-opacity">
                    <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>                      <BarChart data={chartData}>
                      <RechartsTooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #3a3a3a', borderRadius: '8px', fontSize: '12px' }} cursor={{ fill: '#2a2a2a' }} />
                      {hasBivariate ? (
                        // Bivariate Stacked Bars
                        bivariateData.target_classes.map((tClass: string, idx: number) => (
                          <Bar key={tClass} dataKey={tClass} stackId="a" fill={CHART_COLORS[idx % CHART_COLORS.length]} radius={idx === bivariateData.target_classes.length - 1 ? [2, 2, 0, 0] : [0, 0, 0, 0]} />
                        ))
                      ) : (
                        // Standard Univariate Histogram
                        <Bar dataKey="count" fill={isTargetCol ? "#3b82f6" : "#6366f1"} radius={[2, 2, 0, 0]} />
                      )}
                    </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs text-text-secondary border-t border-border-subtle pt-3">
                    {col.type === 'numeric' ? (
                      <>
                        <div><span className="text-text-muted">Mean:</span> {col.mean ?? '---'}</div>
                        <div><span className="text-text-muted">Median:</span> {col.median ?? '---'}</div>
                        <div><span className="text-text-muted">Min:</span> {col.min ?? '---'}</div>
                        <div><span className="text-text-muted">Max:</span> {col.max ?? '---'}</div>
                        <div><span className="text-text-muted">Kurtosis:</span> {col.kurtosis ?? '---'}</div>
                        <div className="flex items-center gap-1">
                          <span className="text-text-muted">Skew:</span>
                          <span className={col.skewness && Math.abs(col.skewness) > 1 ? 'text-accent-warning font-bold' : ''}>
                            {col.skewness ?? '---'} {col.skewness && Math.abs(col.skewness) > 1 && '⚠️'}
                          </span>
                        </div>
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
              <h3 className="text-lg font-medium mb-4 flex items-center gap-2"><Network size={18} className="text-accent-primary" /> Feature Correlation Matrix</h3>
              <div className="p-6 rounded-xl border border-border-subtle bg-background-surface overflow-x-auto">
                <div className="flex items-start justify-between mb-4">
                  <p className="text-sm text-text-secondary">
                    Visual map of linear relationships. <span className="text-indigo-400 font-medium">Blue = Positive</span>, <span className="text-red-400 font-medium">Red = Negative</span>.
                  </p>
                  <p className="text-xs text-accent-warning flex items-center gap-1"><Info size={14} /> Highlighted cells (|r| &gt; 0.85) indicate potential multi-collinearity.</p>
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

          <div className="flex justify-between items-center mt-8 pt-6 border-t border-border-subtle">
            <p className="text-text-muted text-sm">Review your data health before proceeding to ML setup.</p>
            <button onClick={() => setAppPhase(3)} className="flex items-center gap-2 px-8 py-4 bg-accent-primary hover:bg-indigo-600 text-white rounded-lg font-medium transition-all shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)]">
              Proceed to ML Setup <ChevronRight size={20} />
            </button>
          </div>

        </div>
      </div>
    );
  }

  // ==========================================
  // VIEW 1: UPLOAD ZONE
  // ==========================================
  return (
    <div className="min-h-screen flex flex-col items-center pt-24 px-4 relative">
      <Toaster position="bottom-center" toastOptions={{ style: { background: '#1a1a1a', color: '#f5f5f5', border: '1px solid #2a2a2a' } }} />
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4 text-text-primary">KyD<span className="text-accent-primary">.ai</span></h1>
        <p className="text-text-secondary text-lg">Upload your dataset to get started.</p>
      </div>

      <div
        className={`w-full max-w-2xl p-12 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all duration-200 ease-in-out bg-background-surface
          ${isDragging && !isUploading ? 'border-accent-primary bg-background-elevated' : ''}
          ${!isUploading ? 'border-border-subtle hover:border-border-active hover:bg-background-elevated cursor-pointer' : 'border-border-subtle opacity-80 cursor-not-allowed'}
        `}
        onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} onClick={() => !isUploading && fileInputRef.current?.click()}
      >
        <input type="file" accept=".csv" className="hidden" ref={fileInputRef} onChange={handleFileInput} />
        <div className={`p-4 rounded-full mb-4 ${isDragging && !isUploading ? 'bg-accent-primary/20 text-accent-primary' : 'bg-background-primary text-text-muted'}`}>
          {isUploading ? <Loader2 size={40} className="animate-spin text-accent-primary" /> : <UploadCloud size={40} />}
        </div>
        <h3 className="text-xl font-medium mb-2">{isUploading ? 'Uploading & Analyzing...' : isDragging ? 'Drop CSV here...' : 'Drag & Drop your CSV file'}</h3>
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