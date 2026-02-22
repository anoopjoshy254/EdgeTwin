import { Injectable } from '@angular/core';
import { Subject, BehaviorSubject } from 'rxjs';

export interface SensorReading {
  time: string;
  temp: number;
  vib: number;
  pwr: number;
  hum: number;
  status: 'NORMAL' | 'FAILURE';
  confidence: number;
  failure_count: number;
  report: string | null;
}

@Injectable({ providedIn: 'root' })
export class TwinService {
  private ws: WebSocket | null = null;
  private apiBase = 'http://localhost:8000';
  private wsBase = 'ws://localhost:8000';

  reading$ = new Subject<SensorReading>();
  status$ = new BehaviorSubject<'NORMAL' | 'FAILURE' | 'IDLE'>('IDLE');
  connected$ = new BehaviorSubject<boolean>(false);

  connect(machineType: string) {
    this.disconnect();
    const url = `${this.wsBase}/ws/${machineType}`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => this.connected$.next(true);

    this.ws.onmessage = (evt) => {
      const data: SensorReading = JSON.parse(evt.data);
      this.reading$.next(data);
      this.status$.next(data.status);
    };

    this.ws.onclose = () => {
      this.connected$.next(false);
      this.status$.next('IDLE');
    };

    this.ws.onerror = () => this.connected$.next(false);
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
    this.connected$.next(false);
    this.status$.next('IDLE');
  }

  getMachines(): Promise<string[]> {
    return fetch(`${this.apiBase}/api/machines`)
      .then(r => r.json())
      .then(d => d.machines);
  }

  getDevices(): Promise<any[]> {
    return fetch(`${this.apiBase}/api/devices`)
      .then(r => r.json())
      .then(d => d.devices);
  }

  registerDevice(device: { name: string; machine_type: string; serial_number: string; location: string }) {
    return fetch(`${this.apiBase}/api/devices`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(device)
    }).then(r => r.json());
  }

  registerEmail(email: string): Promise<{ success: boolean; email: string }> {
    return fetch(`${this.apiBase}/api/register-email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email })
    }).then(r => r.json());
  }

  getRegisteredEmail(): Promise<{ email: string | null }> {
    return fetch(`${this.apiBase}/api/register-email`).then(r => r.json());
  }

  unregisterEmail(): Promise<{ success: boolean }> {
    return fetch(`${this.apiBase}/api/register-email`, { method: 'DELETE' }).then(r => r.json());
  }
}
