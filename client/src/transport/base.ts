export interface TransportEvent {
  type: string;
  session_id: string;
  payload: Record<string, unknown>;
}

export interface StreamTransport {
  connect(): Promise<void>;
  sendAudio(data: ArrayBuffer): void;
  sendEvent(event: TransportEvent): void;
  onAudio(cb: (data: ArrayBuffer) => void): void;
  onEvent(cb: (event: TransportEvent) => void): void;
  onClose(cb: () => void): void;
  close(): void;
}
