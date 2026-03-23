import { useCallback } from "react";
import { Card, Group, Slider, Stack, Text, Title } from "@mantine/core";
import { IconVolume, IconVolumeOff } from "@tabler/icons-react";
import type { AudioNodes } from "../hooks/useAudioStream.ts";
import { Waveform } from "./Waveform.tsx";

interface AudioMonitorPanelProps {
  audioNodes: AudioNodes;
  sourceLabel: string;
}

function VolumeControl({
  label,
  icon,
  gain,
  analyser,
}: {
  label: string;
  icon: React.ReactNode;
  gain: GainNode | null;
  analyser: AnalyserNode | null;
}) {
  const handleChange = useCallback(
    (value: number) => {
      if (gain) gain.gain.value = value / 100;
    },
    [gain],
  );

  return (
    <Card withBorder p="xs">
      <Stack gap={4}>
        <Group gap="xs">
          {icon}
          <Title order={6}>{label}</Title>
        </Group>
        <Waveform analyser={analyser} height={48} />
        <Group gap="xs" wrap="nowrap">
          <Text size="xs" c="dimmed" w={50}>
            Volume
          </Text>
          <Slider
            min={0}
            max={150}
            defaultValue={100}
            onChange={handleChange}
            style={{ flex: 1 }}
            label={(v) => `${v}%`}
            disabled={!gain}
          />
        </Group>
      </Stack>
    </Card>
  );
}

export function AudioMonitorPanel({ audioNodes, sourceLabel }: AudioMonitorPanelProps) {
  return (
    <Card withBorder p="md">
      <Title order={5} mb="sm">
        Audio Monitor
      </Title>
      <Stack gap="sm">
        <VolumeControl
          label={sourceLabel}
          icon={<IconVolume size={16} />}
          gain={audioNodes.sourceGain}
          analyser={audioNodes.sourceAnalyser}
        />
        <VolumeControl
          label="Translation Output"
          icon={audioNodes.outputGain?.gain.value === 0 ? <IconVolumeOff size={16} /> : <IconVolume size={16} />}
          gain={audioNodes.outputGain}
          analyser={audioNodes.outputAnalyser}
        />
      </Stack>
    </Card>
  );
}
