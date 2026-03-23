import { useCallback, useEffect, useRef } from "react";
import { useMantineColorScheme } from "@mantine/core";

interface WaveformProps {
  analyser: AnalyserNode | null;
  width?: number;
  height?: number;
}

export function Waveform({ analyser, width = 300, height = 60 }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const { colorScheme } = useMantineColorScheme();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);

    const isDark = colorScheme === "dark";
    ctx.fillStyle = isDark ? "#1a1b1e" : "#f8f9fa";
    ctx.fillRect(0, 0, width, height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#339af0";
    ctx.beginPath();

    const sliceWidth = width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * height) / 2;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    ctx.lineTo(width, height / 2);
    ctx.stroke();

    animRef.current = requestAnimationFrame(draw);
  }, [analyser, width, height, colorScheme]);

  useEffect(() => {
    if (!analyser) return;
    animRef.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(animRef.current);
    };
  }, [analyser, draw]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ borderRadius: 4, width: "100%", height }}
    />
  );
}
