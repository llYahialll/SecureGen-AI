import { useEffect, useState } from "react";

const EXAMPLES = {
  python: {
    sql_injection: `from flask import request\n\nusername = request.args.get("username")\nquery = "SELECT * FROM users WHERE name = '" + username + "'"\ncursor.execute(query)`,
    weak_crypto: `import hashlib\n\ndef hash_password(password):\n    return hashlib.md5(password.encode()).hexdigest()`,
    hardcoded_secret: `API_KEY = "sk_test_9412_prod"\nclient = SomeClient(api_key=API_KEY)`,
  },
  javascript: {
    sql_injection: `const user = req.query.user;\nconst query = "SELECT * FROM users WHERE name = '" + user + "'";\ndb.query(query);`,
    weak_crypto: `const crypto = require("crypto");\nconst hash = crypto.createHash("md5").update(password).digest("hex");`,
    hardcoded_secret: `const apiKey = "prod_live_89112";\nconst client = createClient({ apiKey });`,
  },
  java: {
    sql_injection: `String user = request.getParameter("user");\nString query = "SELECT * FROM users WHERE name = '" + user + "'";\nstatement.executeQuery(query);`,
    weak_crypto: `MessageDigest md = MessageDigest.getInstance("MD5");\nbyte[] digest = md.digest(password.getBytes());`,
    hardcoded_secret: `String password = "super-secret-admin";\nconnection = DriverManager.getConnection(url, "admin", password);`,
  },
};

const PATTERNS = {
  sql_injection: [/SELECT.*\+.*user/i, /execute\(.*\+/i, /f['"]SELECT/i, /query\s*=\s*["'`].*SELECT.*\+/i],
  weak_crypto: [/hashlib\.md5/i, /hashlib\.sha1/i, /DES\.new/i, /MODE_ECB/i, /createHash\(["']md5["']\)/i, /MessageDigest\.getInstance\(["']MD5["']\)/i],
  hardcoded_secret: [/password\s*=\s*['"].+['"]/i, /api_key\s*=/i, /API_KEY\s*=/i, /SECRET.*=/i, /token\s*=/i],
};

const META = {
  sql_injection: {
    severity: "CRITICAL",
    color: "bg-red-100 text-red-800 border-red-200",
    title: "SQL Injection Detected (CWE-089)",
    detail: "String-built SQL introduces a direct injection path. Use parameterized queries or prepared statements instead.",
    learn: "Learn about CWE-089 →",
    remediation: `query = "SELECT * FROM users WHERE name = %s"\ncursor.execute(query, (username,))`,
    riskyTokens: ["SELECT", "+", "execute", "query"],
  },
  weak_crypto: {
    severity: "HIGH",
    color: "bg-orange-100 text-orange-800 border-orange-200",
    title: "Weak Crypto Detected (CWE-327)",
    detail: "The snippet relies on a broken or risky cryptographic primitive or mode. Replace it with current, recommended algorithms.",
    learn: "Learn about CWE-327 →",
    remediation: `digest = hashlib.sha256(password.encode()).hexdigest()`,
    riskyTokens: ["md5", "sha1", "DES", "ECB", "MD5"],
  },
  hardcoded_secret: {
    severity: "HIGH",
    color: "bg-amber-100 text-amber-800 border-amber-200",
    title: "Hardcoded Secret Detected (CWE-798)",
    detail: "Credential material appears directly in source code. Move secrets to environment variables or a secure secret manager.",
    learn: "Learn about CWE-798 →",
    remediation: `api_key = os.environ["SERVICE_API_KEY"]\nclient = SomeClient(api_key=api_key)`,
    riskyTokens: ["password", "apiKey", "API_KEY", "SECRET", "token"],
  },
  safe: {
    severity: "SAFE",
    color: "bg-emerald-100 text-emerald-800 border-emerald-200",
    title: "No Primary Pattern Detected",
    detail: "The rule-based demo did not match one of the core SecureGen AI vulnerability classes. A trained model would still analyze deeper context.",
    learn: "Review full secure coding guidance →",
    remediation: "No remediation generated for this snippet.",
    riskyTokens: [],
  },
};

function randomConfidence(min, max) {
  return Number((Math.random() * (max - min) + min).toFixed(2));
}

function analyzeSnippet(code) {
  for (const [label, regexList] of Object.entries(PATTERNS)) {
    const matched = regexList.find((pattern) => pattern.test(code));
    if (matched) {
      return {
        label,
        confidence: randomConfidence(0.72, 0.96),
        ...META[label],
      };
    }
  }

  return {
    label: "safe",
    confidence: randomConfidence(0.88, 0.97),
    ...META.safe,
  };
}

function highlightRiskyTokens(code, riskyTokens) {
  const escaped = riskyTokens
    .filter(Boolean)
    .map((token) => token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));

  if (escaped.length === 0) {
    return code;
  }

  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
  return code.split(pattern).map((part, index) => {
    const risky = riskyTokens.some((token) => token.toLowerCase() === part.toLowerCase());
    if (!risky) {
      return <span key={`${part}-${index}`}>{part}</span>;
    }

    return (
      <mark key={`${part}-${index}`} className="rounded bg-red-200/90 px-1 text-red-950">
        {part}
      </mark>
    );
  });
}

export default function SecureGenAIDemo() {
  const [language, setLanguage] = useState("python");
  const [code, setCode] = useState(EXAMPLES.python.sql_injection);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    sql_injection: 0,
    weak_crypto: 0,
    hardcoded_secret: 0,
    safe: 0,
  });

  useEffect(() => {
    setCode(EXAMPLES[language].sql_injection);
  }, [language]);

  function loadExample(type) {
    setCode(EXAMPLES[language][type]);
  }

  function handleAnalyze() {
    setLoading(true);
    const snapshot = code;

    window.setTimeout(() => {
      const analysis = analyzeSnippet(snapshot);
      const timestamp = new Date().toLocaleString();
      const entry = {
        timestamp,
        preview: snapshot.slice(0, 56).replace(/\n/g, " "),
        verdict: analysis.label,
        confidence: `${Math.round(analysis.confidence * 100)}%`,
      };

      setResult({
        ...analysis,
        code: snapshot,
      });

      setHistory((current) => [entry, ...current].slice(0, 5));
      setStats((current) => ({
        ...current,
        total: current.total + 1,
        [analysis.label]: (current[analysis.label] || 0) + 1,
      }));
      setLoading(false);
    }, 700);
  }

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#07121f_0%,#10253e_45%,#f0f5f9_45%,#f4f8fb_100%)] text-slate-900">
      <header className="border-b border-white/10 bg-slate-950/70 text-white backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-sky-500 to-cyan-300 text-xl shadow-lg shadow-sky-900/30">
              🔐
            </div>
            <div>
              <div className="text-lg font-black tracking-wide">SecureGen AI</div>
              <div className="text-xs uppercase tracking-[0.18em] text-slate-300">Academic Security Demo</div>
            </div>
          </div>
          <nav className="flex gap-5 text-sm font-semibold text-slate-200">
            <a href="#analyze" className="hover:text-cyan-300">Analyze</a>
            <a href="#about" className="hover:text-cyan-300">About</a>
          </nav>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-8">
        <section className="grid gap-6 rounded-[28px] border border-white/10 bg-slate-950/70 p-6 text-white shadow-2xl shadow-slate-950/30 md:grid-cols-[1.15fr_0.85fr]">
          <div>
            <div className="mb-3 inline-flex rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.18em] text-cyan-200">
              Secure Code Generation Risks
            </div>
            <h1 className="max-w-2xl text-4xl font-black leading-tight md:text-5xl">
              Inspect AI-generated code before risky patterns turn into shipped vulnerabilities.
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
              This demo simulates SecureGen AI’s product workflow: paste code, classify the vulnerability type, review the confidence score, and inspect highlighted risky tokens with a remediation snippet.
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.14em] text-slate-400">Training Set</div>
              <div className="mt-2 text-3xl font-black">130</div>
              <div className="mt-2 text-sm text-slate-300">SecurityEval benchmark scenarios</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.14em] text-slate-400">Target Classes</div>
              <div className="mt-2 text-3xl font-black">4</div>
              <div className="mt-2 text-sm text-slate-300">SQLi, secrets, weak crypto, other</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.14em] text-slate-400">Expected Macro-F1</div>
              <div className="mt-2 text-3xl font-black">85%</div>
              <div className="mt-2 text-sm text-slate-300">Cross-validation target from the guide</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.14em] text-slate-400">Explainability</div>
              <div className="mt-2 text-3xl font-black">Attention</div>
              <div className="mt-2 text-sm text-slate-300">Heatmap-inspired token highlighting</div>
            </div>
          </div>
        </section>

        <section id="analyze" className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-xl shadow-slate-200/60">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-2xl font-black text-slate-900">Code Input Panel</h2>
                <p className="mt-1 text-sm text-slate-500">Paste a snippet or load a real vulnerability-style example.</p>
              </div>
              <select
                className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-2 text-sm font-semibold text-slate-700 outline-none focus:border-cyan-500"
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
              >
                <option value="python">Python</option>
                <option value="javascript">JavaScript</option>
                <option value="java">Java</option>
              </select>
            </div>

            <div className="mb-4 flex flex-wrap gap-2">
              <button className="rounded-full bg-red-100 px-4 py-2 text-sm font-bold text-red-800" onClick={() => loadExample("sql_injection")}>
                SQL Injection Example
              </button>
              <button className="rounded-full bg-orange-100 px-4 py-2 text-sm font-bold text-orange-800" onClick={() => loadExample("weak_crypto")}>
                Weak Crypto
              </button>
              <button className="rounded-full bg-amber-100 px-4 py-2 text-sm font-bold text-amber-800" onClick={() => loadExample("hardcoded_secret")}>
                Hardcoded Secret
              </button>
            </div>

            <textarea
              className="min-h-[320px] w-full rounded-3xl border border-slate-200 bg-slate-950 px-5 py-4 font-mono text-sm leading-7 text-slate-100 outline-none focus:border-cyan-400"
              value={code}
              onChange={(event) => setCode(event.target.value)}
              spellCheck={false}
            />

            <div className="mt-5 flex flex-wrap items-center justify-between gap-4">
              <div className="text-sm text-slate-500">The demo uses pattern matching to simulate the SecureGen AI model output.</div>
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-slate-900 to-cyan-700 px-5 py-3 text-sm font-black text-white shadow-lg shadow-cyan-900/20 transition hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {loading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    Analyzing...
                  </>
                ) : (
                  "Analyze Code"
                )}
              </button>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-xl shadow-slate-200/60">
              <h2 className="text-2xl font-black text-slate-900">Result Panel</h2>
              {!result ? (
                <div className="mt-6 rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-8 text-sm leading-7 text-slate-500">
                  Run an analysis to display the risk badge, vulnerability type, confidence bar, token highlights, remediation guidance, and CWE reference link.
                </div>
              ) : (
                <div className="mt-5">
                  <div className={`inline-flex rounded-full border px-4 py-2 text-sm font-black ${result.color}`}>
                    {result.severity}
                  </div>
                  <h3 className="mt-4 text-2xl font-black text-slate-900">{result.title}</h3>
                  <p className="mt-3 text-sm leading-7 text-slate-600">{result.detail}</p>

                  <div className="mt-5">
                    <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-600">
                      <span>Confidence</span>
                      <span>{Math.round(result.confidence * 100)}%</span>
                    </div>
                    <div className="h-3 rounded-full bg-slate-100">
                      <div
                        className="h-3 rounded-full bg-gradient-to-r from-red-500 via-orange-400 to-cyan-500 transition-all duration-700"
                        style={{ width: `${Math.round(result.confidence * 100)}%` }}
                      />
                    </div>
                  </div>

                  <div className="mt-5 rounded-3xl bg-slate-950 p-4">
                    <div className="mb-3 text-xs font-bold uppercase tracking-[0.18em] text-slate-400">
                      Highlighted Tokens
                    </div>
                    <pre className="whitespace-pre-wrap font-mono text-sm leading-7 text-slate-100">
                      {highlightRiskyTokens(result.code, result.riskyTokens)}
                    </pre>
                  </div>

                  <div className="mt-5 rounded-3xl border border-emerald-200 bg-emerald-50 p-4">
                    <div className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-700">
                      Remediation
                    </div>
                    <pre className="mt-3 whitespace-pre-wrap font-mono text-sm leading-7 text-emerald-950">
                      {result.remediation}
                    </pre>
                  </div>

                  <button className="mt-5 text-sm font-black text-cyan-700">
                    {result.learn}
                  </button>
                </div>
              )}
            </div>

            <div className="rounded-[28px] border border-slate-200 bg-white p-6 shadow-xl shadow-slate-200/60">
              <h2 className="text-2xl font-black text-slate-900">History Panel</h2>
              <div className="mt-4 overflow-hidden rounded-2xl border border-slate-200">
                <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                  <thead className="bg-slate-50 text-slate-600">
                    <tr>
                      <th className="px-4 py-3 font-bold">Timestamp</th>
                      <th className="px-4 py-3 font-bold">Snippet Preview</th>
                      <th className="px-4 py-3 font-bold">Verdict</th>
                      <th className="px-4 py-3 font-bold">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {history.length === 0 ? (
                      <tr>
                        <td colSpan={4} className="px-4 py-6 text-slate-500">
                          No analyses yet.
                        </td>
                      </tr>
                    ) : (
                      history.map((entry, index) => (
                        <tr key={`${entry.timestamp}-${index}`}>
                          <td className="px-4 py-3 text-slate-600">{entry.timestamp}</td>
                          <td className="px-4 py-3 font-mono text-xs text-slate-700">{entry.preview}</td>
                          <td className="px-4 py-3 font-semibold text-slate-900">{entry.verdict}</td>
                          <td className="px-4 py-3 text-slate-700">{entry.confidence}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        <section id="about" className="rounded-[28px] border border-white/10 bg-slate-950/75 p-6 text-white shadow-2xl shadow-slate-950/30">
          <h2 className="text-2xl font-black">Stats Bar</h2>
          <div className="mt-5 grid gap-4 md:grid-cols-5">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Total Analyzed</div>
              <div className="mt-2 text-3xl font-black">{stats.total}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-slate-400">SQL Injections</div>
              <div className="mt-2 text-3xl font-black">{stats.sql_injection}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Weak Crypto</div>
              <div className="mt-2 text-3xl font-black">{stats.weak_crypto}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Secrets</div>
              <div className="mt-2 text-3xl font-black">{stats.hardcoded_secret}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Safe</div>
              <div className="mt-2 text-3xl font-black">{stats.safe}</div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
