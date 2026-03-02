import { useEffect, useRef, useState } from "react";
import {
  Button,
  FileButton,
  Group,
  Modal,
  NativeSelect,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
} from "@mantine/core";
import { createSession, fetchPipelines } from "../api/index.ts";
import type { PipelineInfo, Session } from "../api/index.ts";
import { useAudioDevices } from "../hooks/useAudioDevices.ts";

export type AudioSourceType = "mic" | "file";

export interface AudioSource {
  type: AudioSourceType;
  file?: File;
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
    inputDeviceId: string,
    outputDeviceId: string,
    audioSource: AudioSource,
  ) => void;
}) {
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [selectedPipeline, setSelectedPipeline] = useState("");
  const [label, setLabel] = useState("");
  const [inputDevice, setInputDevice] = useState("");
  const [outputDevice, setOutputDevice] = useState("");
  const [creating, setCreating] = useState(false);
  const [sourceType, setSourceType] = useState<AudioSourceType>("mic");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const resetRef = useRef<() => void>(null);
  const { inputs, outputs } = useAudioDevices();

  useEffect(() => {
    if (!opened) return;
    fetchPipelines().then((p) => {
      setPipelines(p);
      if (p.length > 0) setSelectedPipeline(p[0].id);
    });
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
      });
      const source: AudioSource =
        sourceType === "file" && selectedFile
          ? { type: "file", file: selectedFile }
          : { type: "mic" };
      setLabel("");
      setSelectedFile(null);
      resetRef.current?.();
      onCreated(session, inputDevice, outputDevice, source);
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
            { value: "mic", label: "Live Microphone" },
            { value: "file", label: "MP3 File" },
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
        ) : (
          <Group gap="sm">
            <FileButton
              resetRef={resetRef}
              onChange={(f) => setSelectedFile(f)}
              accept="audio/mpeg,audio/mp3,.mp3"
            >
              {(props) => (
                <Button variant="light" {...props}>
                  Choose MP3 file
                </Button>
              )}
            </FileButton>
            <Text size="sm" c="dimmed" style={{ flex: 1 }}>
              {selectedFile ? selectedFile.name : "No file selected"}
            </Text>
          </Group>
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
        <Group justify="flex-end" mt="xs">
          <Button variant="subtle" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={handleCreate}
            loading={creating}
            disabled={!selectedPipeline || (sourceType === "file" && !selectedFile)}
          >
            Start Session
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}
