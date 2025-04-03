import * as React from "react";

export function Card({ children, className }) {
  return <div className={`rounded-lg border bg-white p-4 shadow-sm ${className}`}>{children}</div>;
}

export function CardHeader({ children }) {
  return <div className="border-b pb-2">{children}</div>;
}

export function CardTitle({ children }) {
  return <h2 className="text-lg font-bold">{children}</h2>;
}

export function CardContent({ children }) {
  return <div className="p-2">{children}</div>;
}
