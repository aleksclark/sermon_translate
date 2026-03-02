import { useEffect, useState } from "react";

export interface AudioDevice {
  deviceId: string;
  label: string;
  kind: MediaDeviceKind;
}

export function useAudioDevices() {
  const [devices, setDevices] = useState<AudioDevice[]>([]);

  useEffect(() => {
    async function enumerate() {
      try {
        await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch {
        return;
      }
      const all = await navigator.mediaDevices.enumerateDevices();
      setDevices(
        all
          .filter((d) => d.kind === "audioinput" || d.kind === "audiooutput")
          .map((d) => ({
            deviceId: d.deviceId,
            label: d.label || `${d.kind} (${d.deviceId.slice(0, 8)})`,
            kind: d.kind,
          })),
      );
    }
    enumerate();
  }, []);

  const inputs = devices.filter((d) => d.kind === "audioinput");
  const outputs = devices.filter((d) => d.kind === "audiooutput");
  return { inputs, outputs };
}
