"""Central security layer for external content: URL/SSRF policy, content validation, bounded fetching.

External content is always untrusted. Every outbound document request goes through `safe_fetch.fetch`;
do not add URL or content checks elsewhere."""
