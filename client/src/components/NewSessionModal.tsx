import { useEffect, useState } from "react";
import {
  Button,
  Group,
  Loader,
  Modal,
  NativeSelect,
  NumberInput,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
} from "@mantine/core";
import { createSession, fetchPipelines, fetchSamples } from "../api/index.ts";
import type { PipelineInfo, SampleInfo, Session } from "../api/index.ts";
import { useAudioDevices } from "../hooks/useAudioDevices.ts";

export type AudioSourceType = "mic" | "sample";

export interface AudioSource {
  type: AudioSourceType;
  sampleUrl?: string;
  sampleFilename?: string;
}

export function NewSessionModal({
  opened,
  onClose,
  onCreated,
}: {
  opened: boolean;
  onClose: () => void;
  onCreated: (
    session: Session,
    pipelineInfo: PipelineInfo,
    inputDeviceId: string,
    outputDeviceId: string,
    audioSource: AudioSource,
  ) => void;
}) {
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [samples, setSamples] = useState<SampleInfo[]>([]);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [selectedPipeline, setSelectedPipeline] = useState("");
  const [selectedSample, setSelectedSample] = useState("");
  const [label, setLabel] = useState("");
  const [inputDevice, setInputDevice] = useState("");
  const [outputDevice, setOutputDevice] = useState("");
  const [creating, setCreating] = useState(false);
  const [sourceType, setSourceType] = useState<AudioSourceType>("sample");
  const [audioContextSeconds, setAudioContextSeconds] = useState<number>(0);
  const { inputs, outputs } = useAudioDevices();

  useEffect(() => {
    if (!opened) return;
    fetchPipelines().then((p) => {
      setPipelines(p);
      if (p.length > 0) setSelectedPipeline(p[0].id);
    });
    setLoadingSamples(true);
    fetchSamples()
      .then((s) => {
        setSamples(s);
        if (s.length > 0) setSelectedSample(s[0].url);
      })
      .finally(() => setLoadingSamples(false));
  }, [opened]);

  useEffect(() => {
    if (inputs.length > 0 && !inputDevice) setInputDevice(inputs[0].deviceId);
  }, [inputs]);

  useEffect(() => {
    if (outputs.length > 0 && !outputDevice) setOutputDevice(outputs[0].deviceId);
  }, [outputs]);

  const handleCreate = async () => {
    if (!selectedPipeline) return;
    setCreating(true);
    try {
      const session = await createSession({
        pipeline_id: selectedPipeline,
        label: label || undefined,
        audio_context_seconds: audioContextSeconds || undefined,
      });
      const sample = samples.find((s) => s.url === selectedSample);
      const source: AudioSource =
        sourceType === "sample" && sample
          ? { type: "sample", sampleUrl: sample.url, sampleFilename: sample.filename }
          : { type: "mic" };
      setLabel("");
      onCreated(session, selected!, inputDevice, outputDevice, source);
      onClose();
    } finally {
      setCreating(false);
    }
  };

  const selected = pipelines.find((p) => p.id === selectedPipeline);

  return (
    <Modal opened={opened} onClose={onClose} title="New Session" centered>
      <Stack gap="sm">
        <NativeSelect
          label="Pipeline"
          data={pipelines.map((p) => ({ value: p.id, label: p.name }))}
          value={selectedPipeline}
          onChange={(e) => setSelectedPipeline(e.currentTarget.value)}
        />
        {selected && (
          <Text size="xs" c="dimmed">
            {selected.description}
          </Text>
        )}
        <TextInput
          label="Label (optional)"
          placeholder="My session"
          value={label}
          onChange={(e) => setLabel(e.currentTarget.value)}
        />
        <SegmentedControl
          fullWidth
          data={[
            { value: "sample", label: "Server Sample" },
            { value: "mic", label: "Live Microphone" },
          ]}
          value={sourceType}
          onChange={(v) => setSourceType(v as AudioSourceType)}
        />
        {sourceType === "mic" ? (
          <NativeSelect
            label="Audio Input"
            data={
              inputs.length > 0
                ? inputs.map((d) => ({ value: d.deviceId, label: d.label }))
                : [{ value: "", label: "No devices found" }]
            }
            value={inputDevice}
            onChange={(e) => setInputDevice(e.currentTarget.value)}
          />
        ) : loadingSamples ? (
          <Group justify="center" py="sm">
            <Loader size="sm" />
            <Text size="sm" c="dimmed">Loading samples…</Text>
          </Group>
        ) : samples.length === 0 ? (
          <Text size="sm" c="dimmed">No samples available on server</Text>
        ) : (
          <NativeSelect
            label="Audio Sample"
            data={samples.map((s) => ({ value: s.url, label: s.filename }))}
            value={selectedSample}
            onChange={(e) => setSelectedSample(e.currentTarget.value)}
          />
        )}
        <NativeSelect
          label="Audio Output"
          data={
            outputs.length > 0
              ? outputs.map((d) => ({ value: d.deviceId, label: d.label }))
              : [{ value: "", label: "No devices found" }]
          }
          value={outputDevice}
          onChange={(e) => setOutputDevice(e.currentTarget.value)}
        />
        <NumberInput
          label="Audio Context (seconds)"
          description="Previous audio fed to the model for continuity (0 = none)"
          value={audioContextSeconds}
          onChange={(v) => setAudioContextSeconds(typeof v === "number" ? v : 0)}
          min={0}
          max={30}
          step={1}
          allowDecimal={false}
        />
        <Group justify="flex-end" mt="xs">
          <Button variant="subtle" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={handleCreate}
            loading={creating}
            disabled={!selectedPipeline || (sourceType === "sample" && !selectedSample)}
          >
            Start Session
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}
