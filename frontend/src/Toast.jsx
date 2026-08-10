import { useEffect, useState } from "react";

export default function Toast({ toasts, remove }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast toast-${t.type}`}>
          <span>{t.message}</span>
          <button className="toast-close" onClick={() => remove(t.id)}>×</button>
        </div>
      ))}
    </div>
  );
}

let _id = 0;
export function useToasts() {
  const [toasts, setToasts] = useState([]);

  const add = (message, type = "info") => {
    const id = ++_id;
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000);
  };

  const remove = id => setToasts(prev => prev.filter(t => t.id !== id));

  return { toasts, remove, success: m => add(m, "success"), error: m => add(m, "error"), info: m => add(m, "info") };
}
