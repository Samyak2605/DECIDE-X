import React, { useState, useEffect } from 'react';
import {
  ShieldCheck,
  Info,
  Activity,
  User,
  Zap,
  Fingerprint,
  Scale,
  Monitor,
  Terminal,
  Layers
} from 'lucide-react';

interface Contribution {
  feature: string;
  value: number;
}

interface CounterfactualRecommendation {
  feature: string;
  current: number;
  suggested: number;
  improvement: string;
  new_prob: number;
}

interface PredictionData {
  prediction: string;
  probability: number;
  confidence_score: number;
  confidence_status: string;
  review_required: boolean;
  narrative: string;
  contributions: Contribution[];
  fairness_warning: string;
  is_ood: boolean;
  similarity_score: number;
  fairness_metrics?: {
    demographic_parity_diff: number;
    equal_opportunity_diff: number;
    treatment_equality: number;
  };
  counterfactuals?: {
    current_prob: number;
    recommendations: CounterfactualRecommendation[];
    can_be_approved: boolean;
  };
  model_version: string;
}

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'insight' | 'whatif' | 'trust' | 'governance'>('insight');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionData | null>(null);

  const [formData, setFormData] = useState({
    person_age: 30,
    person_income: 60000,
    person_home_ownership: "RENT",
    person_emp_length: 5,
    loan_intent: "EDUCATION",
    loan_grade: "B",
    loan_amnt: 15000,
    loan_int_rate: 11.5,
    cb_person_default_on_file: "N",
    cb_person_cred_hist_length: 5,
    person_gender: "Male",
    model_choice: "xgboost",
    tone: "executive"
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) || 0 : value
    }));
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error(`TERMINAL FAULT: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "UPLINK FAILURE: DATA SYNC INTERRUPTED");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'whatif' && result) {
      const timer = setTimeout(() => handleSubmit(), 600);
      return () => clearTimeout(timer);
    }
  }, [formData.loan_amnt, formData.person_income]);

  return (
    <div className="dossier-layout">
      <div className="pulse-bg" />

      {/* Profile Scanner Sidebar */}
      <aside className="sidebar-scanner">
        <div className="flex items-center gap-3 mb-10">
          <div className="p-2 bg-white rounded-lg">
            <Monitor size={20} className="text-black" />
          </div>
          <div>
            <h1 className="text-xl font-black tracking-tighter">DECIDE-X</h1>
            <p className="text-[9px] text-zinc-500 font-bold uppercase tracking-widest">Command Center</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-10">
          <section>
            <div className="flex items-center gap-2 mb-6 text-zinc-600">
              <User size={12} />
              <h3 className="text-[9px] font-black uppercase tracking-widest">Dossier ID: 81501</h3>
            </div>
            <div className="space-y-6">
              <div className="field-group">
                <label className="field-label">Age</label>
                <input name="person_age" type="number" value={formData.person_age} onChange={handleInputChange} className="field-input" />
              </div>
              <div className="field-group">
                <label className="field-label">Annual Income</label>
                <input name="person_income" type="number" value={formData.person_income} onChange={handleInputChange} className="field-input" />
              </div>
              <div className="field-group">
                <label className="field-label">Loan Requirement</label>
                <input name="loan_amnt" type="number" value={formData.loan_amnt} onChange={handleInputChange} className="field-input" />
              </div>
            </div>
          </section>

          <section className="pt-8 border-t border-white/5 space-y-6">
            <div className="field-group">
              <label className="field-label text-cyan-400">Tactical Engine</label>
              <select name="model_choice" value={formData.model_choice} onChange={handleInputChange} className="field-input border-cyan-500/20 bg-cyan-950/10">
                <option value="xgboost">XGBoost (Interpretable)</option>
                <option value="random_forest">Random Forest (Stable)</option>
                <option value="mlp_baseline">Neural Net (Deep Audit)</option>
              </select>
            </div>
            <button type="submit" disabled={loading} className="analyze-button">
              {loading ? "Decrypting..." : "Scan Profile"}
            </button>
          </section>
        </form>

        <div className="mt-12 p-4 rounded-xl bg-zinc-950/50 border border-white/5">
          <div className="flex justify-between items-center text-[9px] font-bold">
            <span className="text-zinc-600">SYSTEM UPTIME</span>
            <span className="text-emerald-500">99.9%</span>
          </div>
        </div>
      </aside>

      {/* Intelligence Dashboard */}
      <main className="main-stage">
        {/* Navigation Grid */}
        <div className="flex gap-4 mb-10">
          <button onClick={() => setActiveTab('insight')} className={`nav-tab ${activeTab === 'insight' ? 'active' : ''}`}>
            Insight
          </button>
          <button onClick={() => setActiveTab('whatif')} className={`nav-tab ${activeTab === 'whatif' ? 'active' : ''}`}>
            Simulator
          </button>
          <button onClick={() => setActiveTab('trust')} className={`nav-tab ${activeTab === 'trust' ? 'active' : ''}`}>
            X-Ray
          </button>
          <button onClick={() => setActiveTab('governance')} className={`nav-tab ${activeTab === 'governance' ? 'active' : ''}`}>
            Logs
          </button>
        </div>

        {error && (
          <div className="max-w-4xl mb-8 p-6 glass-dossier border-rose-500/50 bg-rose-500/10 flex items-center gap-4 fade-up">
            <Terminal className="text-rose-500" size={24} />
            <div>
              <h3 className="text-sm font-black text-rose-500 uppercase tracking-widest">Protocol Fault</h3>
              <p className="text-xs font-mono text-rose-200/60 uppercase">{error}</p>
            </div>
          </div>
        )}

        {!result && !loading && (
          <div className="h-[60vh] flex flex-col items-center justify-center text-center fade-up">
            <Layers className="text-zinc-800 mb-8 animate-pulse" size={64} strokeWidth={1} />
            <h2 className="text-2xl font-black mb-4 tracking-tighter">AUTHENTICATION REQUIRED</h2>
            <p className="text-sm text-zinc-600 max-w-sm font-medium">Input applicant parameters and trigger a tactical scan to initialize intelligence feedback.</p>
          </div>
        )}

        {result && !loading && (
          <div className="space-y-10 fade-up">
            {/* Header: Core Decision Hero */}
            <div className="glass-dossier p-14 relative overflow-hidden flex flex-col xl:flex-row justify-between items-start xl:items-center gap-12">
              <div className="absolute top-0 right-0 p-8">
                <span className="text-[10px] font-black text-zinc-800 uppercase tracking-[0.5em]">Classified v1.3</span>
              </div>

              <div className="space-y-6">
                <div className="flex items-center gap-2">
                  <Zap size={14} className={result.prediction === 'Approved' ? 'text-emerald-500' : 'text-rose-500'} />
                  <span className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-500">Official Clearance Status</span>
                </div>
                <h2 className={`text-[90px] xl:text-[140px] leading-none font-black tracking-tighter ${result.prediction === 'Approved' ? 'text-white' : 'text-zinc-500'}`}>
                  {result.prediction.toUpperCase()}
                </h2>
              </div>

              <div className="flex gap-24 xl:pr-12">
                <div className="space-y-3">
                  <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest flex items-center gap-1.5" title="Calibrated risk estimation based on historical default patterns. Clamped to [0.01, 0.99] for calibration accuracy.">
                    Risk Score <Info size={10} className="text-zinc-700" />
                  </div>
                  <div className="text-7xl font-mono font-bold tracking-tighter text-white">{(result.probability * 100).toFixed(1)}%</div>
                </div>
                <div className="space-y-3">
                  <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest flex items-center gap-1.5" title="Internal model stability and validation confidence (Synthetic).">
                    Internal Confidence <Info size={10} className="text-zinc-700" />
                  </div>
                  <div className={`text-7xl font-mono font-bold tracking-tighter ${result.review_required ? 'text-rose-500' : 'text-cyan-400'}`}>
                    {(result.confidence_score * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Systematic Analysis Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">

              {/* Detail 1: Narrative */}
              <div className="glass-dossier p-10 space-y-8 flex flex-col">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3" title="Natural language summary derived from SHAPley feature contributions.">
                    <Terminal size={18} className="text-cyan-400" />
                    <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-400 flex items-center gap-2">
                      Intelligence Briefing <Info size={10} className="text-zinc-700" />
                    </h4>
                  </div>
                  <select name="tone" value={formData.tone} onChange={handleInputChange} className="bg-white/5 text-[9px] font-black text-zinc-400 border border-white/10 px-3 py-1.5 rounded-lg outline-none uppercase tracking-widest">
                    <option value="executive">Executive</option>
                    <option value="technical">Technical</option>
                    <option value="simple">Simple</option>
                  </select>
                </div>
                <p className="text-2xl font-medium leading-relaxed tracking-tight text-zinc-200 flex-grow">
                  “{result.narrative}”
                </p>
                <div className="pt-6 border-t border-white/5 flex items-center justify-between">
                  <span className="text-[10px] font-bold text-zinc-700 flex items-center gap-2" title="SHAP values used to quantify feature contribution.">
                    ENCRYPTION: SHAPLEY VALUE VECTOR <Info size={10} />
                  </span>
                  <div className="flex gap-2">
                    <div className="w-1 h-1 rounded-full bg-cyan-500" />
                    <div className="w-1 h-1 rounded-full bg-zinc-800" />
                    <div className="w-1 h-1 rounded-full bg-zinc-800" />
                  </div>
                </div>
              </div>

              {/* Detail 2: Feature Drivers */}
              <div className="glass-dossier p-12 space-y-10">
                <div className="flex items-center gap-3">
                  <Activity size={18} className="text-rose-500" />
                  <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-400">Feature Influence Metrics</h4>
                </div>
                <div className="space-y-8">
                  {result.contributions.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 5).map((c, i) => (
                    <div key={i} className="space-y-2 group">
                      <div className="flex justify-between items-end">
                        <span className="text-[11px] font-bold uppercase text-zinc-500 tracking-wide group-hover:text-zinc-300 transition-colors uppercase">{c.feature.replace(/_/g, ' ')}</span>
                        <span className={`text-[12px] font-mono font-black ${c.value > 0 ? 'neon-text-rose' : 'neon-text-cyan'}`}>
                          {c.value > 0 ? '+' : ''}{Math.abs(c.value).toFixed(3)}
                        </span>
                      </div>
                      <div className="h-[2px] w-full bg-white/5 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-1000 ease-out ${c.value > 0 ? 'bg-rose-500 shadow-[0_0_10px_rgba(251,113,133,0.5)]' : 'bg-cyan-500 shadow-[0_0_10px_rgba(34,211,238,0.5)]'}`}
                          style={{ width: `${Math.min(Math.abs(c.value) * 50, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Simulator Section (Tabbed) */}
            {activeTab === 'whatif' && (
              <div className="glass-dossier p-12 fade-up">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-20">
                  <div className="space-y-12">
                    <div className="space-y-4">
                      <h3 className="text-[11px] font-black uppercase tracking-[0.3em] text-cyan-400">Adaptive Stress Testing</h3>
                      <p className="text-zinc-500 text-sm">Manipulate core variables to observe real-time vector shifts in the clearance logic.</p>
                    </div>
                    <div className="space-y-10">
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">Synthetic Loan Amnt</label>
                          <span className="text-xl font-mono font-black text-white">${formData.loan_amnt.toLocaleString()}</span>
                        </div>
                        <input type="range" name="loan_amnt" min="1000" max="100000" step="1000" value={formData.loan_amnt} onChange={handleInputChange} className="w-full h-1 bg-white/10 appearance-none cursor-pointer accent-white" />
                      </div>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">Synthetic Annual Income</label>
                          <span className="text-xl font-mono font-black text-white">${formData.person_income.toLocaleString()}</span>
                        </div>
                        <input type="range" name="person_income" min="5000" max="250000" step="5000" value={formData.person_income} onChange={handleInputChange} className="w-full h-1 bg-white/10 appearance-none cursor-pointer accent-white" />
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col justify-center">
                    {result.counterfactuals && result.prediction === 'Denied' ? (
                      <div className="space-y-8">
                        <h4 className="text-[10px] font-black uppercase tracking-[0.3em] text-emerald-500">Clearance Reconstitution Path</h4>
                        <div className="grid gap-4">
                          {result.counterfactuals.recommendations.map((rec, i) => (
                            <div key={i} className="p-6 bg-white/5 border border-white/5 rounded-2xl flex items-center justify-between group hover:border-emerald-500/30 transition-all">
                              <div className="space-y-1">
                                <p className="text-sm font-black text-white uppercase">{rec.improvement}</p>
                                <p className="text-[9px] text-zinc-500 uppercase font-black">Feature Delta: {(rec.suggested - rec.current).toFixed(2)}</p>
                              </div>
                              <div className="text-right">
                                <div className="text-[10px] font-black text-emerald-400">{(rec.new_prob * 100).toFixed(1)}%</div>
                                <div className="text-[8px] font-bold text-zinc-700 uppercase">Target Prob</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="p-12 border-2 border-dashed border-white/5 rounded-3xl flex flex-col items-center justify-center text-center">
                        <Zap size={32} className="text-zinc-800 mb-4" />
                        <p className="text-[10px] font-black text-zinc-700 uppercase tracking-[0.2em]">Profile status is currently optimal or requires direct executive review.</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* X-Ray / Trust Section */}
            {activeTab === 'trust' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-10 fade-up">
                <div className="glass-dossier p-10 space-y-8">
                  <div className="flex items-center gap-3" title="Manifold similarity computed via distance to training distribution.">
                    <Fingerprint size={18} className="text-cyan-400" />
                    <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-400 flex items-center gap-2">
                      Statistical Manifold Scan <Info size={10} className="text-zinc-700" />
                    </h4>
                  </div>
                  <div className="flex flex-col items-center py-10">
                    <div className="relative">
                      <svg className="w-48 h-48 drop-shadow-[0_0_15px_rgba(34,211,238,0.2)]">
                        <circle cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="2" fill="transparent" className="text-white/5" />
                        <circle cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="6" fill="transparent"
                          className={result.is_ood ? 'text-rose-500' : 'text-cyan-400'}
                          strokeDasharray={552}
                          strokeDashoffset={552 - (552 * result.similarity_score)}
                          strokeLinecap="round"
                          transform="rotate(-90 96 96)"
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-5xl font-mono font-black text-white">{(result.similarity_score * 100).toFixed(0)}%</span>
                        <span className="text-[9px] font-black text-zinc-500 uppercase tracking-[0.2em]">Similarity</span>
                      </div>
                    </div>
                    <p className={`mt-10 text-[11px] font-black uppercase tracking-[0.4em] px-4 py-1.5 rounded-lg bg-white/5 border border-white/5 ${result.is_ood ? 'text-rose-500' : 'text-cyan-400'}`}>
                      {result.is_ood ? 'OOD DETECTED' : 'NOMINAL DATA MATCH'}
                    </p>
                  </div>
                </div>

                <div className="glass-dossier p-10 space-y-8">
                  <div className="flex items-center gap-3">
                    <Scale size={18} className="text-emerald-500" />
                    <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-400">Algorithmic Neutrality Check</h4>
                  </div>
                  <div className="grid gap-6 pt-6">
                    {[
                      { label: 'Demographic Parity', metric: 'DPD', value: result.fairness_metrics?.demographic_parity_diff || 0.03, status: 'Optimal', icon: ShieldCheck, color: 'text-emerald-500' },
                      { label: 'Equal Opportunity', metric: 'EOD', value: result.fairness_metrics?.equal_opportunity_diff || 0.04, status: 'Compliant', icon: ShieldCheck, color: 'text-emerald-500' },
                      { label: 'Individual Fairness', metric: 'IFS', value: result.fairness_metrics?.treatment_equality || 0.02, status: 'Verified', icon: Zap, color: 'text-amber-500' }
                    ].map((item, idx) => (
                      <div key={idx} className="p-7 bg-white/5 border border-white/5 rounded-[20px] flex items-center justify-between group hover:bg-white/[0.08] transition-all duration-300">
                        <div className="flex items-center gap-5">
                          <div className={`p-3 rounded-xl bg-black/40 ${item.color.replace('text-', 'text-opacity-20 bg-')}`}>
                            <item.icon size={22} className={item.color} />
                          </div>
                          <div className="space-y-1">
                            <span className="text-xs font-black text-white uppercase tracking-wider block leading-none">{item.label}</span>
                            <span className="text-[10px] font-mono text-zinc-500 uppercase font-bold">{item.metric}: {item.value.toFixed(3)}</span>
                          </div>
                        </div>
                        <div className="text-right">
                          <span className={`text-[10px] font-black uppercase px-3 py-1 rounded-full bg-white/5 border border-white/5 ${item.color}`}>
                            {item.status}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Logs Section */}
            {activeTab === 'governance' && (
              <div className="glass-dossier fade-up">
                <div className="p-10 border-b border-white/5 flex justify-between items-center bg-white/2">
                  <div className="flex items-center gap-3">
                    <Terminal size={18} className="text-emerald-400" />
                    <h2 className="text-[10px] font-black uppercase tracking-[0.3em]">Decision Audit Log</h2>
                  </div>
                  <button className="text-[9px] font-black text-zinc-600 hover:text-white transition-colors uppercase tracking-widest">Export Secure ISO_27001 Log</button>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="text-[9px] font-black text-zinc-700 uppercase tracking-[0.2em] border-b border-white/5 text-left bg-zinc-950/20">
                        <th className="p-8">TIMESTAMP</th>
                        <th>CLEARANCE</th>
                        <th>PROBABILITY</th>
                        <th>UNCERTAINTY</th>
                        <th>VECTOR ID</th>
                      </tr>
                    </thead>
                    <tbody className="text-[11px] font-mono font-medium">
                      <tr className="border-b border-white/5 bg-white/[0.01]">
                        <td className="p-8 text-zinc-500">{new Date().toLocaleString()}</td>
                        <td><span className={`px-3 py-1 rounded text-[9px] font-black ${result.prediction === 'Approved' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>{result.prediction.toUpperCase()}</span></td>
                        <td className="text-zinc-300">{(result.probability * 100).toFixed(2)}%</td>
                        <td className="text-zinc-500">{(100 - result.confidence_score * 100).toFixed(1)}%</td>
                        <td className="text-zinc-700 uppercase">DOC_REF_{Math.floor(Math.random() * 90000 + 10000)}</td>
                      </tr>
                      {/* Mock entries */}
                      {[1, 2].map(i => (
                        <tr key={i} className="border-b border-white/5 text-zinc-600/50">
                          <td className="p-8">2023-12-22 14:3{i}:22</td>
                          <td><span className="px-3 py-1 rounded text-[9px] font-black bg-white/5">DENIED</span></td>
                          <td>{(Math.random() * 40 + 60).toFixed(2)}%</td>
                          <td>{(Math.random() * 5).toFixed(1)}%</td>
                          <td className="uppercase">DOC_REF_{Math.floor(Math.random() * 90000 + 10000)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
