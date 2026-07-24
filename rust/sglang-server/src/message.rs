//! Messages moved between stages via `flume` (zero-copy moves); variable-length
//! buffers are `bytes::Bytes`, so egress fan-out to detok shards is a refcount bump.
//! Grouped by flow direction: [`request`] (the `/generate` body fan-out, the
//! in-flight request bodies + scheduler ingress wire), [`egress`]
//! (the response back-channel + egress-ring frames and decoded chunk events),
//! [`sampling`] (sampling-params normalization, the Python `SamplingParams` port).
#![allow(dead_code)] // TODO: remove when the consumer PR lands

mod request;
