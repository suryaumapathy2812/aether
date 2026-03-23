FROM caddy:2-builder AS builder

WORKDIR /src

COPY caddy-cloudflare ./caddy-cloudflare

RUN xcaddy build \
    --with github.com/caddy-dns/cloudflare=./caddy-cloudflare \
    && mv ./caddy /usr/bin/caddy

FROM caddy:2

COPY --from=builder /usr/bin/caddy /usr/bin/caddy
