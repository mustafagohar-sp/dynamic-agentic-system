"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

export default function Home() {
  const [status, setStatus] = useState("Checking backend...");

  useEffect(() => {
    getHealth()
      .then((data) => {
        setStatus(data.status);
      })
      .catch(() => {
        setStatus("Backend unavailable");
      });
  }, []);

  return (
    <main className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <h1 className="text-3xl font-bold">
          Dynamic Agentic System
        </h1>

        <p className="mt-4">
          Backend status:{" "}
          <span className="font-semibold">{status}</span>
        </p>
      </div>
    </main>
  );
}