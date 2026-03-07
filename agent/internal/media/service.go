package media

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"path/filepath"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"
	"github.com/google/uuid"
	agentcfg "github.com/suryaumapathy2812/core-ai/agent/internal/config"
)

type Service struct {
	bucket         string
	bucketTemplate string
	client         *s3.Client
	presign        *s3.PresignClient
	publicPresign  *s3.PresignClient // signs URLs using publicBase so browsers can reach them
	putTTL         time.Duration
	getTTL         time.Duration
	forcePath      bool
	publicBase     string
	endpoint       string
	defaultCDN     bool
}

type PutURL struct {
	ObjectKey string
	UploadURL string
	Headers   map[string]string
	ExpiresAt time.Time
}

type ObjectInfo struct {
	Key         string
	Size        int64
	ContentType string
	ETag        string
}

// New creates a media Service from centralized config.
// Returns (nil, nil) if no bucket is configured.
func New(ctx context.Context, cfg agentcfg.S3Config) (*Service, error) {
	bucket := strings.TrimSpace(cfg.Bucket)
	bucketTemplate := strings.TrimSpace(cfg.BucketTemplate)
	if bucket == "" && bucketTemplate == "" {
		return nil, nil
	}
	region := strings.TrimSpace(cfg.Region)
	if region == "" {
		region = "us-east-1"
	}

	accessKey := strings.TrimSpace(cfg.AccessKeyID)
	secretKey := strings.TrimSpace(cfg.SecretAccessKey)
	endpoint := strings.TrimSpace(cfg.Endpoint)
	publicBase := strings.TrimRight(strings.TrimSpace(cfg.PublicBaseURL), "/")
	forcePath := cfg.ForcePathStyle
	putTTL := cfg.PutURLTTL
	getTTL := cfg.GetURLTTL

	loadOpts := []func(*awsconfig.LoadOptions) error{awsconfig.WithRegion(region)}
	if accessKey != "" && secretKey != "" {
		loadOpts = append(loadOpts, awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(accessKey, secretKey, "")))
	}
	if endpoint != "" {
		resolver := s3.EndpointResolverFromURL(endpoint)
		loadOpts = append(loadOpts, awsconfig.WithEndpointResolverWithOptions(aws.EndpointResolverWithOptionsFunc(func(service, region string, options ...any) (aws.Endpoint, error) {
			if service == s3.ServiceID {
				return resolver.ResolveEndpoint(region, s3.EndpointResolverOptions{})
			}
			return aws.Endpoint{}, &aws.EndpointNotFoundError{}
		})))
	}
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return nil, err
	}
	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		o.UsePathStyle = forcePath
	})
	presign := s3.NewPresignClient(client)

	// When a public base URL is set and differs from the internal endpoint,
	// create a second presign client whose endpoint is the public URL.  This
	// ensures the Host header baked into the AWS v4 signature matches the host
	// the browser will actually send the request to.
	var publicPresign *s3.PresignClient
	if publicBase != "" && endpoint != "" && publicBase != strings.TrimRight(endpoint, "/") {
		pubLoadOpts := []func(*awsconfig.LoadOptions) error{awsconfig.WithRegion(region)}
		if accessKey != "" && secretKey != "" {
			pubLoadOpts = append(pubLoadOpts, awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(accessKey, secretKey, "")))
		}
		pubResolver := s3.EndpointResolverFromURL(publicBase)
		pubLoadOpts = append(pubLoadOpts, awsconfig.WithEndpointResolverWithOptions(aws.EndpointResolverWithOptionsFunc(func(service, reg string, options ...any) (aws.Endpoint, error) {
			if service == s3.ServiceID {
				return pubResolver.ResolveEndpoint(reg, s3.EndpointResolverOptions{})
			}
			return aws.Endpoint{}, &aws.EndpointNotFoundError{}
		})))
		pubCfg, pubErr := awsconfig.LoadDefaultConfig(ctx, pubLoadOpts...)
		if pubErr == nil {
			pubClient := s3.NewFromConfig(pubCfg, func(o *s3.Options) {
				o.UsePathStyle = forcePath
			})
			publicPresign = s3.NewPresignClient(pubClient)
		}
	}

	return &Service{
		bucket:         bucket,
		bucketTemplate: bucketTemplate,
		client:         client,
		presign:        presign,
		publicPresign:  publicPresign,
		putTTL:         putTTL,
		getTTL:         getTTL,
		forcePath:      forcePath,
		publicBase:     publicBase,
		endpoint:       strings.TrimRight(endpoint, "/"),
		defaultCDN:     publicBase != "",
	}, nil
}

func (s *Service) Enabled() bool {
	if s == nil || s.client == nil {
		return false
	}
	return strings.TrimSpace(s.bucket) != "" || strings.TrimSpace(s.bucketTemplate) != ""
}

func (s *Service) BucketForUser(userID string) string {
	if s == nil {
		return ""
	}
	if strings.TrimSpace(s.bucketTemplate) == "" {
		return strings.TrimSpace(s.bucket)
	}
	user := sanitizeBucketSegment(userID)
	if user == "" {
		user = "default"
	}
	tmpl := s.bucketTemplate
	if strings.Contains(tmpl, "{user}") {
		return normalizeBucketName(strings.ReplaceAll(tmpl, "{user}", user))
	}
	return normalizeBucketName(tmpl + "-" + user)
}

func (s *Service) BuildObjectKey(userID, sessionID, fileName string) string {
	ext := strings.ToLower(strings.TrimSpace(filepath.Ext(fileName)))
	if len(ext) > 12 {
		ext = ""
	}
	session := sanitizeSegment(sessionID)
	if session == "" {
		session = "chat"
	}
	now := time.Now().UTC()
	if strings.TrimSpace(s.bucketTemplate) != "" {
		return fmt.Sprintf("sessions/%s/%04d/%02d/%02d/%s%s", session, now.Year(), now.Month(), now.Day(), uuid.NewString(), ext)
	}
	user := sanitizeSegment(userID)
	if user == "" {
		user = "default"
	}
	return fmt.Sprintf("users/%s/sessions/%s/%04d/%02d/%02d/%s%s", user, session, now.Year(), now.Month(), now.Day(), uuid.NewString(), ext)
}

func (s *Service) PresignUpload(ctx context.Context, bucket, objectKey, contentType string) (PutURL, error) {
	if !s.Enabled() {
		return PutURL{}, errors.New("media storage is not configured")
	}
	if strings.TrimSpace(objectKey) == "" {
		return PutURL{}, errors.New("object key is required")
	}
	in := &s3.PutObjectInput{
		Bucket: aws.String(strings.TrimSpace(bucket)),
		Key:    aws.String(objectKey),
	}
	if strings.TrimSpace(contentType) != "" {
		in.ContentType = aws.String(contentType)
	}
	// Use the public presign client when available so the signature's Host
	// header matches the browser-accessible URL (e.g. localhost:9000 instead
	// of the Docker-internal minio:9000).
	signer := s.presign
	if s.publicPresign != nil {
		signer = s.publicPresign
	}
	res, err := signer.PresignPutObject(ctx, in, s3.WithPresignExpires(s.putTTL))
	if err != nil {
		return PutURL{}, err
	}
	headers := map[string]string{}
	for k, vals := range res.SignedHeader {
		if len(vals) > 0 {
			headers[strings.ToLower(k)] = vals[0]
		}
	}
	if strings.TrimSpace(contentType) != "" {
		headers["content-type"] = contentType
	}
	return PutURL{
		ObjectKey: objectKey,
		UploadURL: res.URL,
		Headers:   headers,
		ExpiresAt: time.Now().UTC().Add(s.putTTL),
	}, nil
}

func (s *Service) EnsureBucket(ctx context.Context, bucket string) error {
	bucket = strings.TrimSpace(bucket)
	if bucket == "" {
		return errors.New("bucket is required")
	}
	_, err := s.client.HeadBucket(ctx, &s3.HeadBucketInput{Bucket: aws.String(bucket)})
	if err == nil {
		// Bucket exists — ensure the public-read policy is applied (idempotent).
		_ = s.ensurePublicReadPolicy(ctx, bucket)
		return nil
	}
	// Some production IAM policies do not allow HeadBucket/ListBucket but still
	// allow object-level Put/Get with presigned URLs. In that case we should not
	// fail media init; skip bucket management and continue.
	if isAccessDenied(err) {
		return nil
	}
	_, err = s.client.CreateBucket(ctx, &s3.CreateBucketInput{Bucket: aws.String(bucket)})
	if err != nil {
		var apiErr smithy.APIError
		if errors.As(err, &apiErr) {
			code := strings.TrimSpace(apiErr.ErrorCode())
			if code != "BucketAlreadyOwnedByYou" && code != "BucketAlreadyExists" {
				return err
			}
		} else {
			return err
		}
	}
	// Apply public-read policy so browsers can fetch objects directly via
	// the public base URL without presigned query parameters.
	_ = s.ensurePublicReadPolicy(ctx, bucket)
	return nil
}

// ensurePublicReadPolicy sets an anonymous read-only policy on the bucket.
// This allows browsers to GET objects from the public base URL without
// presigned credentials.  The policy is idempotent.
func (s *Service) ensurePublicReadPolicy(ctx context.Context, bucket string) error {
	policy := fmt.Sprintf(`{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::%s/*"]
    }
  ]
}`, bucket)
	_, err := s.client.PutBucketPolicy(ctx, &s3.PutBucketPolicyInput{
		Bucket: aws.String(bucket),
		Policy: aws.String(policy),
	})
	return err
}

func (s *Service) HeadObject(ctx context.Context, bucket, objectKey string) (ObjectInfo, error) {
	out, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{Bucket: aws.String(strings.TrimSpace(bucket)), Key: aws.String(objectKey)})
	if err != nil {
		return ObjectInfo{}, err
	}
	info := ObjectInfo{Key: objectKey}
	if out.ContentLength != nil {
		info.Size = *out.ContentLength
	}
	if out.ContentType != nil {
		info.ContentType = strings.TrimSpace(*out.ContentType)
	}
	if out.ETag != nil {
		info.ETag = strings.Trim(*out.ETag, "\"")
	}
	return info, nil
}

func (s *Service) PresignGet(ctx context.Context, bucket, objectKey string) (string, error) {
	if !s.Enabled() {
		return "", errors.New("media storage is not configured")
	}
	if strings.TrimSpace(objectKey) == "" {
		return "", errors.New("object key is required")
	}
	bucket = strings.TrimSpace(bucket)
	if s.defaultCDN && s.forcePath && bucket != "" {
		// Path-style S3 (e.g. MinIO): public URL includes the bucket segment.
		return s.publicBase + "/" + bucket + "/" + strings.TrimPrefix(objectKey, "/"), nil
	}
	if s.defaultCDN {
		// Virtual-hosted or CDN: bucket is implied by the domain.
		return s.publicBase + "/" + strings.TrimPrefix(objectKey, "/"), nil
	}
	res, err := s.presign.PresignGetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(bucket), Key: aws.String(objectKey)}, s3.WithPresignExpires(s.getTTL))
	if err != nil {
		return "", err
	}
	return res.URL, nil
}

func (s *Service) GetObjectBytes(ctx context.Context, bucket, objectKey string) ([]byte, string, error) {
	out, err := s.client.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(strings.TrimSpace(bucket)), Key: aws.String(objectKey)})
	if err != nil {
		return nil, "", err
	}
	defer out.Body.Close()
	b, err := io.ReadAll(out.Body)
	if err != nil {
		return nil, "", err
	}
	ct := ""
	if out.ContentType != nil {
		ct = strings.TrimSpace(*out.ContentType)
	}
	return b, ct, nil
}

func ExtensionFromMime(m string) string {
	m = strings.TrimSpace(strings.ToLower(m))
	if m == "" {
		return ""
	}
	exts, _ := mime.ExtensionsByType(m)
	if len(exts) == 0 {
		return ""
	}
	return strings.ToLower(exts[0])
}

func sanitizeSegment(v string) string {
	v = strings.TrimSpace(strings.ToLower(v))
	if v == "" {
		return ""
	}
	out := make([]rune, 0, len(v))
	for _, r := range v {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			out = append(out, r)
		}
	}
	if len(out) == 0 {
		return ""
	}
	return string(out)
}

func sanitizeBucketSegment(v string) string {
	v = strings.TrimSpace(strings.ToLower(v))
	if v == "" {
		return ""
	}
	out := make([]rune, 0, len(v))
	for _, r := range v {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			out = append(out, r)
		}
	}
	if len(out) == 0 {
		return ""
	}
	return string(out)
}

func normalizeBucketName(v string) string {
	v = strings.Trim(strings.ToLower(strings.TrimSpace(v)), "-.")
	if len(v) < 3 {
		v = v + "-media"
	}
	if len(v) > 63 {
		v = v[:63]
		v = strings.Trim(v, "-.")
	}
	return v
}

func IsNotFound(err error) bool {
	var nsk *types.NotFound
	return errors.As(err, &nsk)
}

func isAccessDenied(err error) bool {
	var apiErr smithy.APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	code := strings.TrimSpace(strings.ToLower(apiErr.ErrorCode()))
	switch code {
	case "accessdenied", "forbidden", "unauthorized", "unauthorizedoperation":
		return true
	default:
		return false
	}
}
